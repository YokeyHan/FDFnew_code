import copy
import torch.nn as nn
import torch.nn.functional as F
import math
from models.generate_adj import *
import numpy as np
from einops import rearrange
import torch

def Attn_tem(heads=8, layers=1, channels=8):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=8, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer_QKV(nn.Module):  # 此处与CSDI相同
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True, device=None, is_adp=True, adj_file=None):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, base_shape, support_adp):
        B, channel, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec1 = support_adp[-1][0]
            nodevec2 = support_adp[-1][1]
            support = support_adp[:-1]
        else:
            support = support_adp
        x = x.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        if self.is_adp:
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            support = support + [adp]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out


class TemporalLearning000(nn.Module):
    def __init__(self, channels, nheads, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None):   #[B, channel, N, L]->
        B, channel, N, L = base_shape    #B,self.channels,L,N
        if L == 1:
            return y

        y = y.reshape(B, channel, N, L).permute(0, 2, 1, 3).reshape(B * N, channel, L)   #[B, channel, N, L]->[B, N, channel, L]->[B*N, channel, L]  ->[L, B*N, channel]
        #y = y.reshape(B, L, N).permute(1, 0, 2)   #[B, L, N]-> ->[L, B, N]
        #v = y
        v = y.permute(2, 0, 1)
        if self.is_cross:  # Attention weights:False与CSDI相同
            itp_y = itp_y.reshape(B, channel, N, L).permute(0, 2, 1, 3).reshape(B * N, channel, L)   #[B, channel, N, L]->[B, N, channel, L]->[B*N, channel, L]  ->[L, B*N, channel]
            q = itp_y.permute(2, 0, 1)  #[B * N, channel, L]->[L, B*N, channel]
            y = self.time_layer(q, q, v).permute(1, 2, 0)   #[L, B*N, channel]->[B*N, L, channel]
            #itp_y = itp_y.reshape(B, L, N).permute(1, 0, 2)
            #q = itp_y
            y = self.time_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, N, channel, L).permute(0, 2, 1, 3).reshape(B, channel, N * L)
        #y = y.reshape(B, L, N)
        return y


class FeatureLearning000(nn.Module):
    def __init__(self, channels, nheads, is_cross=True):
        super().__init__()
        self.is_cross = is_cross

        self.feature_layer = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, N, L = base_shape   #[B,channel,N * L]
        if L == 1:
            return y
        y = y.reshape(B, channel, N, L).permute(0, 3, 1, 2).reshape(B * L, channel, N)   #[B, channel, K, L]->[B, L, channel, K]->[B*L, channel, K]  ->[K, B*L, channel]
        v = y.permute(2, 0, 1)  # (K, B * L, channel)
        if self.is_cross:  # Attention weights:False与CSDI相同
            itp_y = itp_y.reshape(B, channel, N, L).permute(0, 3, 1, 2).reshape(B * L, channel, N)
            q = itp_y.permute(2, 0, 1)
            y = self.feature_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.feature_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, L, channel, N).permute(0, 2, 3, 1).reshape(B, channel, N * L)
        return y


class FeatureLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t, is_cross):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = FeaDependLearning(channels, nheads=nheads, order=order, target_dim=target_dim,
                                               include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file,
                                               proj_t=proj_t, is_cross=is_cross)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class FeaDependLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t,
                 is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp,
                               adj_file=adj_file)
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)  # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        # y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # if self.is_cross:
        #    itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        #    y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        # else:
        #    y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        # y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

        # y_attn = y_in1 + y_attn
        # y_attn = self.norm1_attn(y_attn)

        # y_in2 = y_local + y_attn
        # y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))

        y = F.relu(self.ff_linear1(y_local.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_local

        y = self.norm2(y)
        return y


class SpatialLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t, is_cross):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = SpaDependLearning(channels, nheads=nheads, order=order, target_dim=target_dim,
                                               include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file,
                                               proj_t=proj_t, is_cross=is_cross)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class SpaDependLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t,
                 is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp,
                               adj_file=adj_file)
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)  # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

        y_attn = y_in1 + y_attn
        y_attn = self.norm1_attn(y_attn)

        y_in2 = y_local + y_attn
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


class GuidanceConstruct(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t):
        super().__init__()
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp,
                               adj_file=adj_file)
        self.attn_s = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.attn_t = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn_s = nn.GroupNorm(4, channels)
        self.norm1_attn_t = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)  # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)

        y_attn_s1 = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y_attn_s = self.attn_s(y_attn_s1.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn_s = y_attn_s.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn_s = y_in1 + y_attn_s
        y_attn_s = self.norm1_attn_s(y_attn_s)

        y_attn_t1 = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y_attn_t1.permute(2, 0, 1)
        y_attn_t = self.attn_t(v, v, v).permute(1, 2, 0)
        y_attn_t = y_attn_t.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        y_attn_t = y_in1 + y_attn_t
        y_attn_t = self.norm1_attn_t(y_attn_t)

        y_in2 = y_local + y_attn_s + y_attn_t
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Attn_spa(nn.Module):  # 实现了一种可缩放的点积注意力机制: 此注意机制类似于Transformer模型中使用的机制，但进行了修改，将键和值沿着序列长度维度投影到大小为k的较低维空间(这是模块的超参数)
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class SpatemDecoding(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, device, is_adp, adj_file, proj_t, is_cross):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = SpatemLearning(channels, nheads=nheads, order=order, target_dim=target_dim,
                                            include_self=True, device=device, is_adp=is_adp, adj_file=adj_file,
                                            proj_t=proj_t, is_cross=is_cross)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class SpatemLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t,
                 is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp,
                               adj_file=adj_file)
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        y_in1 = y.reshape(B, channel, K * L)

        y_local = self.GCN(y.reshape(B, channel, K * L), base_shape, support)  # [B, C, K*L]

        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        # y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # if self.is_cross:
        #    itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        #    y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        # else:
        #    y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        # y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

        # y_attn = y_in1 + y_attn
        # y_attn = self.norm1_attn(y_attn)

        # y_in2 = y_local + y_attn
        # y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))

        # y = F.relu(self.ff_linear1(y_local.permute(0, 2, 1)))
        # y = self.ff_linear2(y).permute(0, 2, 1)
        # y = y + y_local

        # y = self.norm2(y)
        return y_local


def kmeans_plus_plus_init(data, k, device='cpu'):
    """
    K-means++ 初始化方法

    参数:
    data: 输入数据 [样本数, 特征维度]
    k: 聚类中心数量
    device: 计算设备

    返回:
    centers: 初始化的聚类中心 [k, 特征维度]
    """
    # 使用detach()移除梯度要求，然后转换为numpy
    data = data.detach().cpu().numpy()  # 转换为numpy进行计算
    n_samples, n_features = data.shape
    centers = np.zeros((k, n_features))

    # 随机选择第一个中心
    center_id = np.random.randint(n_samples)
    centers[0] = data[center_id]

    # 计算每个样本到最近中心的距离的平方
    distances = np.sum((data - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        # 按照距离的平方作为权重，随机选择下一个中心
        probs = distances / np.sum(distances)
        next_center_id = np.random.choice(n_samples, p=probs)
        centers[i] = data[next_center_id]

        # 更新每个样本到最近中心的距离
        if i < k - 1:
            new_distances = np.sum((data - centers[i]) ** 2, axis=1)
            distances = np.minimum(distances, new_distances)

    return torch.tensor(centers, dtype=torch.float32, device=device)


class TemporalLearning(nn.Module):
    def __init__(self, channels, nheads, is_cross_t):
        super().__init__()
        self.layers = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nheads,
            dim_feedforward=channels * 4,
            batch_first=True
        )

    def forward(self, x, base_shape):
        B, C, N, L = base_shape
        x = rearrange(x, 'B C N L -> (B N) L C')
        x = self.layers(x)
        return rearrange(x, '(B N) L C -> B C N L', B=B, N=N)


class FeatureLearning0(nn.Module):
    def __init__(self, channels, nheads, is_cross_s):
        super().__init__()
        self.layers = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nheads,
            dim_feedforward=channels * 4,
            batch_first=True
        )

    def forward(self, x, base_shape, adj):
        B, C, N, L = base_shape
        x = rearrange(x, 'B C N L -> (B L) N C')
        x = self.layers(x)
        return rearrange(x, '(B L) N C -> B C N L', B=B, L=L)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)


        attention = torch.where(adj[0] > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Create spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(x_cat))
        return x * attention



class SpatialVectorQuantizer0(nn.Module):
    def __init__(self, num_embeddings, batch_size, seq_len, commitment_cost, decay=0.99):
        super(SpatialVectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings  # n_cluster
        self._batch_size = batch_size  # B
        self._seq_len = seq_len  # L
        self._embedding_dim = batch_size * seq_len  # B * L
        self._commitment_cost = commitment_cost
        # EMA parameters
        self.decay = decay

        # 码本形状: [num_embeddings, B * L]
        # 每个码本条目代表一个在时间-批次维度上的模式
        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        # Buffers for EMA updates - these are not model parameters
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs):
        # 输入形状: [B, L, z_dim*N]
        # 我们需要将其转换为可以与码本匹配的形状
        B, L, spatial_feature_dim = inputs.shape  # spatial_feature_dim = z_dim * N

        # 转换输入以匹配码本维度
        # [B, L, z_dim*N] -> [z_dim*N, B*L] (转置以便每行是一个空间特征在所有时间步的表示)
        inputs_transposed = inputs.permute(2, 0, 1).reshape(spatial_feature_dim, -1)  # [z_dim*N, B*L]

        # 对每个空间特征维度进行量化
        quantized_features = []
        encoding_indices_list = []
        losses = []
        perplexities = []

        for i in range(spatial_feature_dim):
            feature_vector = inputs_transposed[i].unsqueeze(0)  # [1, B*L]

            aa=feature_vector.unsqueeze(1)   # [1, B*L]
            ab=self.embedding.weight.unsqueeze(0)

            # 计算与码本的距离
            distances = torch.sum((feature_vector.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2,
                                  dim=2)  # [1, num_embeddings]

            # 找到最近的码本条目
            encoding_indices = torch.argmin(distances, dim=1)  # [1]
            encodings = F.one_hot(encoding_indices, num_classes=self._num_embeddings).float()  # [1, num_embeddings]

            # 量化
            quantized_feature = torch.matmul(encodings, self.embedding.weight)  # [1, B*L]

            # 计算损失
            e_latent_loss = F.mse_loss(quantized_feature.detach(), feature_vector)
            q_latent_loss = F.mse_loss(quantized_feature, feature_vector.detach())
            feature_loss = q_latent_loss + self._commitment_cost * e_latent_loss

            # Straight-through estimator
            quantized_feature = feature_vector + (quantized_feature - feature_vector).detach()

            quantized_features.append(quantized_feature)
            encoding_indices_list.append(encoding_indices)
            losses.append(feature_loss)

            # 计算困惑度
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            perplexities.append(perplexity)

        # 重新组合量化结果
        quantized_transposed = torch.cat(quantized_features, dim=0)  # [z_dim*N, B*L]
        quantized_output = quantized_transposed.reshape(spatial_feature_dim, B, L).permute(1, 2, 0)  # [B, L, z_dim*N]

        # 合并损失和困惑度
        total_loss = torch.mean(torch.stack(losses))
        avg_perplexity = torch.mean(torch.stack(perplexities))

        # 编码索引
        all_encodings = torch.stack(encoding_indices_list, dim=0)  # [z_dim*N, 1]

        return total_loss, quantized_output, avg_perplexity, all_encodings.squeeze(1)


class SpatialVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, batch_size, seq_len, commitment_cost, decay=0.99):
        super(SpatialVectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings  # n_cluster
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._embedding_dim = batch_size * seq_len  # B * L
        self._commitment_cost = commitment_cost

        # EMA parameters
        self.decay = decay

        # Initialize the codebook (embedding) - 维度为 [num_embeddings, B*L]
        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        # Buffers for EMA updates - these are not model parameters
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs):
        # 输入形状: [B, L, z_dim*N]
        B, L, spatial_feature_dim = inputs.shape  # spatial_feature_dim = z_dim * N

        # 转换输入形状以便进行量化
        # [B, L, z_dim*N] -> [z_dim*N, B*L]
        inputs_transposed = inputs.permute(2, 0, 1).reshape(spatial_feature_dim, -1)  # [z_dim*N, B*L]

        # 对每个空间特征维度独立进行量化
        quantized_features = []
        encoding_indices_list = []
        losses = []
        all_encodings = []

        for i in range(spatial_feature_dim):
            feature_vector = inputs_transposed[i].unsqueeze(0)  # [1, B*L]

            # 计算与码本的距离
            distances = torch.sum((feature_vector.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2,
                                  dim=2)  # [1, num_embeddings]

            # 找到最近的码本条目
            encoding_indices = torch.argmin(distances, dim=1)  # [1]
            encodings = F.one_hot(encoding_indices, num_classes=self._num_embeddings).float()  # [1, num_embeddings]

            # 量化
            quantized_feature = torch.matmul(encodings, self.embedding.weight)  # [1, B*L]

            # EMA更新 (仅在训练时)
            if self.training:
                with torch.no_grad():
                    # 更新簇大小EMA
                    self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                             (1 - self.decay) * encodings.squeeze(0)

                    # 更新码本向量EMA
                    dw = torch.matmul(encodings.t(), feature_vector)  # [num_embeddings, B*L]
                    self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw

                    # Laplace平滑避免零计数
                    n = torch.sum(self._ema_cluster_size)
                    self._ema_cluster_size = (
                            (self._ema_cluster_size + 1e-5)
                            / (n + self._num_embeddings * 1e-5) * n
                    )

                    # 标准化更新的码本向量
                    self.embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))

            # 计算损失
            commitment_loss = F.mse_loss(feature_vector, quantized_feature.detach())
            feature_loss = self._commitment_cost * commitment_loss

            # 直通估计器
            quantized_feature = feature_vector + (quantized_feature - feature_vector).detach()

            quantized_features.append(quantized_feature)
            encoding_indices_list.append(encoding_indices)
            losses.append(feature_loss)
            all_encodings.append(encodings.squeeze(0))

        # 重新组合量化结果
        quantized_transposed = torch.cat(quantized_features, dim=0)  # [z_dim*N, B*L]
        quantized_output = quantized_transposed.reshape(spatial_feature_dim, B, L).permute(1, 2, 0)  # [B, L, z_dim*N]

        # 合并损失
        total_loss = torch.mean(torch.stack(losses))

        # 计算困惑度
        all_encodings_tensor = torch.stack(all_encodings, dim=0)  # [z_dim*N, num_embeddings]
        avg_probs = torch.mean(all_encodings_tensor, dim=0)  # [num_embeddings]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # 编码索引
        all_indices = torch.stack(encoding_indices_list, dim=0)  # [z_dim*N, 1]

        return total_loss, quantized_output, perplexity, all_indices.squeeze(1)
