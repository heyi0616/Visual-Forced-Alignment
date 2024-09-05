from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence

from models.resnet_new import ResNetBackBone, BasicBlock
from models.conformer_xattn import ConformerCrossAttn
from data import consts
from utils.sequence_decoder import greedy_decode


class Visual_front(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super().__init__()

        self.in_channels = in_channels
        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNetBackBone(BasicBlock, [2, 2, 2, 2], backend_out=out_channels)

    def forward(self, x):
        x = self.frontend(x)    # B,C,T,H,W
        B, C, T, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B*T, C, H, W)
        x = self.resnet(x)  # B*T, 512             #0.20 sec (5 frames)
        x = x.view(B, T, -1)   # B, T, 512
        return x


class ConformerCrossAttnEncoder(nn.Module):
    def __init__(self, input_dim, num_layer=4):
        super().__init__()
        self.pos_enc = PositionalEncoding(input_dim)
        self.encoder = ConformerCrossAttn(dim=input_dim, depth=num_layer, dim_head=64, heads=8)

    def forward(self, x, token=None, frame_mask=None, label_mask=None):
        x = self.pos_enc(x)
        x = self.encoder(x, token, frame_mask, label_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)  # (max_len, 1, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding, batch_first=False):
        if batch_first:
            token_embedding = token_embedding.permute(1, 0, 2)
        pos_enc = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        if batch_first:
            pos_enc = pos_enc.permute(1, 0, 2)
        return pos_enc


class MultiLinear(nn.Module):
    def __init__(self, d_in, d_out, d_inner):
        super(MultiLinear, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_inner),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d_inner, d_out)
        )

    def forward(self, x):
        return self.net(x)


class PhonemeSplitDecoderNet(nn.Module):
    def __init__(self, num_class):
        super(PhonemeSplitDecoderNet, self).__init__()
        embed_dim = consts.EMBED_SIZE
        self.frontend = Visual_front(in_channels=1, out_channels=embed_dim)
        hidden_dim = consts.EMBED_SIZE

        self.pad = 0
        self.ph_embed = nn.Embedding(num_class, embedding_dim=embed_dim, padding_idx=self.pad)
        self.ph_sentence_pred = ConformerCrossAttnEncoder(embed_dim, num_layer=4)
        self.frame_linear = MultiLinear(hidden_dim, num_class, 128)
        # self.ph_linear = MultiLinear(hidden_dim, num_class, 128)
        self.ph_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden_dim, 8), num_layers=2)
        self.ph_linear = nn.Linear(hidden_dim, num_class)
        self.ph_w_sil_embed = nn.Embedding(num_class, embedding_dim=embed_dim, padding_idx=self.pad)
        self.pos_embed = PositionalEncoding(embed_dim, maxlen=800)

        self.bound_linear = MultiLinear(hidden_dim, 1, 128)

    def forward(self, x, label, label_mask=None, frame_mask=None, ph_label_w_sil=None):
        x = self.frontend(x)
        x = self.pos_embed(x, batch_first=True)
        token = self.ph_embed(label)
        token = self.pos_embed(token, batch_first=True)
        x = self.ph_sentence_pred(x, token=token, frame_mask=frame_mask, label_mask=label_mask)
        x_out = self.frame_linear(x)
        tgt_mask = self.generate_square_subsequent_mask(ph_label_w_sil.shape[1]-1).cuda()
        token_w_sil = self.ph_w_sil_embed(ph_label_w_sil)
        label_w_sil_mask = ph_label_w_sil != 0
        B = x.shape[0]
        x_filtered = [x[b, frame_mask[b]] for b in range(B)]
        label_filtered = [token[b, label_mask[b]] for b in range(B)]
        xy_concat = [torch.cat([yf, xf], dim=0) for yf, xf in zip(label_filtered, x_filtered)]
        padded_xy = pad_sequence(xy_concat, batch_first=True, padding_value=0)
        new_mask = [torch.cat([torch.ones(yf.size(0), dtype=torch.bool), torch.ones(xf.size(0), dtype=torch.bool)]) for
                    yf, xf in zip(label_filtered, x_filtered)]
        new_mask_padded = pad_sequence(new_mask, batch_first=True, padding_value=0).cuda()
        label_out = self.ph_decoder(self.pos_embed(token_w_sil[:, :-1].permute(1, 0, 2)), padded_xy.permute(1, 0, 2), tgt_mask=tgt_mask, tgt_key_padding_mask=~label_w_sil_mask[:, :-1],
                                    memory_key_padding_mask=~new_mask_padded)
        label_out = self.ph_linear(label_out.permute(1, 0, 2))
        bound_out = self.bound_linear(x).squeeze(-1)
        return x_out, label_out, bound_out

    def predict(self, x, label, label_mask=None, frame_mask=None):
        x = self.frontend(x)
        x = self.pos_embed(x, batch_first=True)
        token = self.ph_embed(label)
        token = self.pos_embed(token, batch_first=True)
        x = self.ph_sentence_pred(x, token=token, frame_mask=frame_mask, label_mask=label_mask)
        x_out = self.frame_linear(x)
        B = x.shape[0]
        x_filtered = [x[b, frame_mask[b]] for b in range(B)]
        label_filtered = [token[b, label_mask[b]] for b in range(B)]
        xy_concat = [torch.cat([yf, xf], dim=0) for yf, xf in zip(label_filtered, x_filtered)]
        preds = []
        for i in range(B):
            memory = xy_concat[i].unsqueeze(1)
            pred = greedy_decode(self, self.ph_w_sil_embed, memory, consts.TOKEN_MAX_LEN, linear=self.ph_linear, beam_width=8)
            preds.append(pred)
        bound_out = self.bound_linear(x).squeeze(-1)
        bound_out = F.sigmoid(bound_out)
        return x_out, preds, bound_out

    def decode(self, tgt_embed, memory, tgt_mask):
        return self.ph_decoder(self.pos_embed(tgt_embed), memory, tgt_mask=tgt_mask)

    def generate_square_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask