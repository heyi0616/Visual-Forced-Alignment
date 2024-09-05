import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


# from https://github.com/lucidrains/conformer

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            max_pos_emb=512,
            local_attn_win=0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)
        self.local_attn_win = local_attn_win

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device=device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(
                *context.shape[:2], device=device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            if self.local_attn_win > 0:
                local_mask = self.gen_local_mask(x.shape[0], n, window_size=self.local_attn_win).cuda()
                mask = (mask * local_mask).bool()
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

    def gen_local_mask(self, batch_size, seq_length, window_size=16):
        mask = torch.zeros([batch_size, 1, seq_length, seq_length])
        for i in range(seq_length):
            start = max(0, i - window_size)
            end = min(seq_length, i + window_size + 1)
            mask[:, :, i, start:end] = 1
        return mask


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0.,
            conv_causal=False,
            local_attn_win=0,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, local_attn_win=0)
        self.attn = PreNorm(dim, self.attn)
        self.conv = ConformerConvModule(dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn2 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, local_attn_win=local_attn_win)
        self.attn2 = PreNorm(dim, self.attn2)
        self.conv2 = ConformerConvModule(dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor,
                                         kernel_size=3, dropout=conv_dropout)  # local kernel=3
        self.cross_attn = MultiHeadAttention(d_model=dim, n_head=heads, d_k=dim // heads, d_v=dim // heads,
                                             dropout=attn_dropout)

        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, token, frame_mask=None, token_mask=None):
        x = self.ff1(x) + x
        cross_mask = torch.bmm(frame_mask.unsqueeze(-1).float(), token_mask.unsqueeze(1).float()).int()
        x = self.cross_attn(x, token, mask=cross_mask)
        self_attn1 = self.attn(x, mask=frame_mask)+x
        self_attn2 = self.attn2(x, mask=frame_mask)+x
        x = torch.maximum(self.conv(self_attn1) + self_attn1, self.conv2(self_attn2) + self_attn2)
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


# Conformer

class ConformerCrossAttn(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0.,
            conv_causal=False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        local_attn_windows = [16, 16, 16, 16]
        for i in range(depth):
            self.layers.append(ConformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                conv_causal=conv_causal,
                local_attn_win=local_attn_windows[i]
            ))

    def forward(self, x, token, frame_mask=None, token_mask=None):

        for block in self.layers:
            x = block(x, token, frame_mask, token_mask)

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_kv = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, kv, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), kv.size(1), kv.size(1)

        residual = q
        q = self.layer_norm(q)
        kv = self.layer_norm(kv)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(kv).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(kv).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        #
        # q = self.layer_norm(q)
        return q


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q,k,v (b x n_head x len_q x dv)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # (b, n_head, len_q, len_v)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
