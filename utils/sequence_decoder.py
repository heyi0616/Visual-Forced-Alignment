import torch
import torch.nn.functional as F

from data import consts


def greedy_decode(model, tgt_embed_ins, memory, max_len, **kwargs):
    """
    :param model: Transformer
    :param tgt_embed_ins: target TokenEmbedding
    :param memory: (seq_len, 1, embed_size)
    :param max_len:
    :return: (n, 1)
    """
    ys = torch.full((1, 1), consts.BOS).type(torch.long).cuda()
    for i in range(max_len - 1):
        memory = memory.cuda()
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).cuda()
        tgt_embed = tgt_embed_ins(ys)
        out = model.decode(tgt_embed, memory, tgt_mask)
        out = out.reshape(-1, out.shape[2])  # (seq_len, embed_size)
        if kwargs.get("linear", None):
            prob = kwargs["linear"](out[-1, :])
        else:
            prob = torch.matmul(out[-1, :], torch.transpose(tgt_embed_ins.weight, 0, 1))  # (vocab_size)

        next_word = torch.argmax(prob, -1).squeeze().item()
        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).cuda()], dim=0)
        if next_word == consts.EOS:
            break
    # print("greedy out size: " + str(ys.squeeze().shape))
    return ys.squeeze()


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
    return mask
