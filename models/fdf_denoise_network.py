import math
import torch
from torch import nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from models.ChebyKANLayer import *
from models.layers import *


#维度变换 封装张量的维度置换操作，常用于调整CNN和Transformer的维度顺序。
class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)

#正弦位置编码  为时间步（如扩散模型的噪声步）生成固定模式的位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#自适应层归一化 根据时间步动态调整层归一化的参数，用于扩散模型的条件控制
class AdaLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb = SinusoidalPosEmb(emb_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_dim, emb_dim*2)
        self.layernorm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        
    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


# 修改后的Conv_MLP，适应四维输入[B, L, N, z_dim]
class myConv_MLP(nn.Module):
    def __init__(self, z_dim, emb_dim, N, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 3)),  # [B, z_dim, N, L]
            nn.Conv2d(z_dim, emb_dim, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Transpose(shape=(1, 3)),  # [B, L, N, emb_dim]
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        #[B,L,N,z_dim]
        return self.sequential(x)

#卷积+MLP模块 将输入通过1D卷积映射到高维空间，用于特征提取。
class Conv_MLP(nn.Module):
    def __init__(self, in_dim, emb_dim, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, emb_dim, 3, stride=1, padding=1),
            nn.Dropout(p=dropout),
        )
    def forward(self, x):
        return self.sequential(x).transpose(1, 2)

# 修改后的LearnablePositionalEncoding，适应四维输入[B, L, N, emb_dim]
class myLearnablePositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 为时间维度和节点维度分别创建位置编码
        self.time_pe = nn.Parameter(torch.empty(1, max_len, 1, emb_dim))
        self.node_pe = nn.Parameter(torch.empty(1, 1, 1024, emb_dim))  # 假设最大节点数为1024
        nn.init.uniform_(self.time_pe, -0.02, 0.02)
        nn.init.uniform_(self.node_pe, -0.02, 0.02)

    def forward(self, x):
        """
        x: [B, L, N, emb_dim]
        """
        # 根据输入的节点数截断位置编码
        node_pe = self.node_pe[:, :, :x.size(2), :]
        x = x + self.time_pe[:, :x.size(1), :, :] + node_pe
        return self.dropout(x)

#可学习位置编码 替代固定位置编码，让模型自动学习序列位置信息。
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, emb_dim)) 
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class LINEAR(nn.Module):
    def __init__(self, in_features, out_features, drop=0.1):
        super(LINEAR, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features,order=4):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            in_features,
                            out_features,
                            order)
    def forward(self, x):
        B, L,N, C = x.shape   #[B, L, N, emb_dim]
        x = self.fc1(x.reshape(B*L*N,C))
        x = x.reshape(B,L,N,-1).contiguous()
        return x

# 修改后的STDecoderBlock，适应四维输入[B, L, N, emb_dim]
class mySTDecoderBlock(nn.Module):
    def __init__(self,
                 n_channel,
                 adj,
                 channels,
                 z_dim,
                 device,
                 emb_dim=96,
                 dropout=0.2,
                 nheads=4,
                 use_temporal=True,
                 use_spatial=True,
                 is_cross_t=False, is_cross_s=False):
        super().__init__()

        self.adj = adj
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial

        # 时间和空间注意力
        if use_spatial:
            #self.spatial_attn = SpatialAttention(emb_dim, adj, dropout)
            #channels, z_dim, num_clusters, adj, device, nheads = 4, is_cross_t = False, is_cross_s = False, inputdim = 1
            self.spatial_attn = FeatureLearning000(channels=channels, nheads=nheads, is_cross=is_cross_s)#FeatureLearning(channels=channels, nheads=nheads, target_dim=36,
                                 #              order=2, include_self=True, device=device, is_adp=True,
                                  #             adj_file=adj, proj_t=64,is_cross=is_cross_s)

        if use_temporal:
            self.temporal_attn = TemporalLearning000(channels=channels, nheads=nheads, is_cross=is_cross_t)

        # 条件归一化 - 调整为处理四维输入
        self.ln1 = AdaLayerNorm(emb_dim)

        # 前馈网络
        self.LayerNorm = nn.LayerNorm(emb_dim)
        self.LINEAR = ChebyKANLayer(channels, channels)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)

    def forward(self, x, timestep, mask=None, label_emb=None):
        """
        x: [B, L, N, emb_dim] - 批次大小，时间步，节点数，嵌入维度
        """
        B, L, N, _ = x.shape
        base_shape = x.permute(0, 3, 2,1).shape  #[B,channel,N, L]
        # 条件归一化 - 重塑为 [B, L*N, emb_dim]
        x_reshaped = x.reshape(B, L * N, -1)
        x_norm = self.ln1(x_reshaped, timestep, label_emb)
        # 重塑回 [B, L, N, emb_dim]
        x_norm = x_norm.reshape(B, L, N, -1)

        res1 = x_norm.clone()

        # 调整顺序：先时间注意力，后空间注意力
        if self.use_temporal:
            # 时间注意力处理
            x_temporal = self.temporal_attn(x_norm.permute(0, 3, 2,1).reshape(B,-1,N*L),base_shape)  #[B,C,N*L]  #.permute(0, 3, 2,1)
        else:
            x_temporal = x_norm.permute(0, 3, 2,1).reshape(B,-1,N*L)

        if self.use_spatial:
            # 空间注意力处理 - 调整维度顺序以适应空间注意力模块
            x_permuted = x_temporal  # [B, N, L, emb_dim]
            x_spatial = self.spatial_attn(x_permuted,base_shape, self.adj)  #[B,C,N*L]
            # 调整回原始维度顺序
            x_attended = x_spatial.reshape(B,-1,N,L).permute(0, 3, 2, 1)  # [B, L, N, emb_dim]
        else:
            x_attended = x_temporal.reshape(B,-1,N,L).permute(0, 3, 2, 1)


        # 应用LINEAR层
        x_linear = self.LINEAR(x_attended)  #[B, L, N, emb_dim]

        res2 = x_linear + res1

        # 前馈网络
        # 重塑为 [B*L*N, emb_dim]
        x_ff = res2.reshape(B * L * N, -1)
        x_ff = self.activation(self.linear1(x_ff))
        x_ff = self.dropout(x_ff)
        x_ff = self.linear2(x_ff)
        # 重塑回 [B, L, N, emb_dim]
        x_ff = x_ff.reshape(B, L, N, -1)

        x_out = x_ff + res2
        x_out = self.LayerNorm(x_out.reshape(B, L * N, -1)).reshape(B, L, N, -1)

        return x_out

class STDecoderBlock(nn.Module):
    def __init__(self,
                 n_channel,
                 adj,
                 emb_dim=96,
                 dropout=0.2,
                 use_temporal=True,
                 use_spatial=True):
        super().__init__()

        self.adj = adj
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial

        # 时间和空间注意力
        if use_spatial:
            #self.spatial_attn = SpatialAttention(emb_dim, adj, dropout)
            self.spatial_attn = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)

        if use_temporal:
            self.temporal_attn = TemporalAttention(emb_dim, dropout)

        # 条件归一化
        self.ln1 = AdaLayerNorm(emb_dim)

        # 前馈网络
        self.LayerNorm = nn.LayerNorm(emb_dim)
        self.LINEAR = LINEAR(n_channel, n_channel)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)

    def forward(self, x, timestep, mask=None, label_emb=None):
        """x [B,pred_len,emb_dim]
        x: [B, N, T, emb_dim] - 批次大小，节点数，时间步，嵌入维度
        """
        B, N, T, _ = x.shape

        # 条件归一化
        # 重塑为 [B, N*T, emb_dim] 以适应原始归一化层
        x_reshaped = x.reshape(B, N * T, -1)
        x_norm = self.ln1(x_reshaped, timestep, label_emb)
        # 重塑回 [B, N, T, emb_dim]
        x_norm = x_norm.reshape(B, N, T, -1)

        res1 = x_norm.clone()

        # 时空注意力处理
        if self.use_temporal:
            x_temporal = self.temporal_attn(x_norm)
        else:
            x_temporal = x_norm

        if self.use_spatial:
            x_spatial = self.spatial_attn(x_temporal)
        else:
            x_spatial = x_temporal


        # 应用原始的LINEAR层
        # 重塑为 [B, T, N, emb_dim] 以便处理
        x_linear = x_spatial.permute(0, 2, 1, 3).reshape(B, T, N, -1)
        x_linear = self.LINEAR(x_linear.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # 重塑回 [B, N, T, emb_dim]
        x_linear = x_linear.reshape(B, N, T, -1)

        res2 = x_linear + res1

        # 前馈网络
        # 重塑为 [B*N*T, emb_dim]
        x_ff = res2.reshape(B * N * T, -1)
        x_ff = self.activation(self.linear1(x_ff))
        x_ff = self.dropout(x_ff)
        x_ff = self.linear2(x_ff)
        # 重塑回 [B, N, T, emb_dim]
        x_ff = x_ff.reshape(B, N, T, -1)

        x_out = x_ff + res2
        x_out = self.LayerNorm(x_out.reshape(B, N * T, -1)).reshape(B, N, T, -1)

        return x_out


#解码器块 扩散模型的核心解码块，融合时间步条件并逐步重建数据
class DecoderBlock(nn.Module):
    def __init__(self,
                 n_channel,
                 emb_dim=96,
                 dropout=0.1
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(emb_dim)
        
        self.LayerNorm = nn.LayerNorm(emb_dim)
        self.LINEAR = LINEAR(n_channel, n_channel)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)

    def forward(self, x, timestep, mask=None, label_emb=None):
        #x [B,pred_len,emb_dim]  timestep 4 [25,42,36,18]
        x = self.ln1(x, timestep, label_emb)
        res1 = x.clone()
        x = self.LINEAR(x.permute(0, 2, 1)).permute(0, 2, 1) 
        res2 = x + res1   #[B, pred_len, emb_dim]
        x = self.activation(self.linear1(res2))     # linear1将维度从 [B, pred_len, emb_dim] 变为 [B, pred_len, emb_dim*4] # 经过SiLU激活函数，维度不变: [B, pred_len, emb_dim*4]
        x = self.dropout(x) 
        x = self.linear2(x)   # linear2将维度从 [B, pred_len, emb_dim*4] 变回 [B, pred_len, emb_dim]
        x = x + res2
        x = self.LayerNorm(x)   # 层归一化，维度不变: [B, pred_len, emb_dim]

        return x 

#多层解码器堆叠 堆叠多个DecoderBlock，逐步去噪生成预测结果
class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        adj,
        channels,
        z_dim,
        device,
        emb_dim=1024,
        n_layer=14,
        dropout=0.1
    ):
      super().__init__()
      self.adj = adj
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[
            nn.Sequential(
                mySTDecoderBlock(
                    n_channel=n_channel,
                    emb_dim=emb_dim,
                    adj=adj,
                    channels=channels,
                    z_dim=z_dim,
                    device=device,
                    dropout=dropout
                ),
                nn.LayerNorm(emb_dim) 
            ) for _ in range(n_layer)
        ])
      
    def forward(self, enc, t, padding_masks=None, label_emb=None):
        x = enc
        for block_idx in range(len(self.blocks)):
            x = self.blocks[block_idx][0](x, t, mask=padding_masks, label_emb=label_emb)
            x = self.blocks[block_idx][1](x)

        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
#扩散去噪网络  条件生成路径：基于历史数据（cond）预测未来序列的均值和方差，通过重参数化生成初步结果。条件控制：通过AdaLayerNorm将时间步和标签信息注入模型。
# 扩散路径：通过解码器对噪声输入（input）逐步去噪。
# 加权融合：结合两种生成方式的结果，平衡确定性和随机性。
class fdf_denoise_network(nn.Module):
    def __init__(
        self,
        n_feat,  #207
        seq_len,
        pred_len,
        device,
        adj,
        channels,
        z_dim,
        MLP_hidden_dim=256,
        emb_dim=1024,
        patch_len=4,
        n_layer=14,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        #self.emb = Conv_MLP(n_feat, emb_dim, dropout=dropout)
        self.adj = adj
        self.z_dim = z_dim  # 计算每个节点的特征维度
        self.N = n_feat // z_dim
        self.emb = myConv_MLP(z_dim, emb_dim, self.N, dropout)
        
        self.sparsity_threshold = 0.01
        
        self.seq_length = seq_len
        self.pred_length = pred_len
        self.patch_len = patch_len
        
        self.device = device

        self.pos_dec = myLearnablePositionalEncoding(emb_dim, dropout=dropout, max_len=pred_len)
        self.decoder = Decoder(self.pred_length, n_feat, self.adj, channels,
                               z_dim, device, emb_dim, n_layer, dropout)
        #self.decoder = Decoder(self.pred_length , n_feat, emb_dim, n_layer, dropout)
        self.feat_linear = nn.Linear(emb_dim, n_feat, bias = True)
        
        self.weight = nn.Parameter(torch.randn(1)) 
        
        self.MLP_hidden_size = MLP_hidden_dim
        
        self.mean_linear = nn.Sequential(
            nn.Linear(self.seq_length // self.patch_len, self.MLP_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.MLP_hidden_size, self.pred_length // self.patch_len)
        )
        self.var_linear = nn.Sequential(
            nn.Linear(self.seq_length // self.patch_len , self.MLP_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.MLP_hidden_size, self.pred_length // self.patch_len)
        )
    
    def forward(self, input, t, cond, padding_masks=None):
        
        batch_size, L, feature_dim = cond.shape  #[B,Pre_len,N]  [B,L,N*z_dim]
        input = input.reshape(batch_size, L, self.N, -1)  # [B,L,N,z_dim]
        num_patches = self.seq_length // self.patch_len
        cond_patches = cond[:, :num_patches * self.patch_len, :].view(batch_size, num_patches, self.patch_len, feature_dim)
        patch_mean = cond_patches.mean(dim=2)   #[B,num_patch,patch_len,feature_dim]-->[B,num_patch,feature_dim]
        patch_var = cond_patches.var(dim=2, unbiased=False).sqrt()  #[B,num_patch,feature_dim]
        pred_mean = self.mean_linear(patch_mean.permute(0, 2, 1)).permute(0, 2, 1)   #[B,feature_dim,num_patch]
        pred_var = self.var_linear(patch_var.permute(0, 2, 1)).permute(0, 2, 1)   #
        pred_patches = self.pred_length // self.patch_len   #
        epsilon = torch.randn(batch_size, pred_patches, self.patch_len, feature_dim, device=cond.device)  #[B,pred_patches, patch_len, feature_dim]
        sampled = pred_mean.unsqueeze(2).repeat(1, 1, self.patch_len, 1) + epsilon * (pred_var).unsqueeze(2).repeat(1, 1, self.patch_len, 1)  #[B,pred_patches, patch_len, feature_dim]
        sampled = sampled.reshape(batch_size, self.patch_len * pred_patches, feature_dim)  #[B,pred_patches*patch_len, feature_dim]
        
        total_var = pred_var.sum(dim=1, keepdim=True)    #[B,1,feature_dim]
        var_ratio = (pred_var / total_var) * 0.5    #[B,pred_patches,feature_dim]
        emb = self.emb(input)
        inp_dec = self.pos_dec(emb)
        output = self.decoder(inp_dec, t, padding_masks=padding_masks)  #[B,pred_len,emb_dim]
        output = output.reshape(batch_size, self.pred_length, -1)

        #output = self.feat_linear(output)
        if output.shape[-1] != sampled.shape[-1]:
            # 如果需要，使用线性变换调整维度
            output = self.feat_linear(output.reshape(batch_size * self.pred_length, -1))
            output = output.reshape(batch_size, self.pred_length, -1)

        result = self.weight * output + (1 - self.weight) * sampled   # [B, seq_len, n_feat]
        
        #result = self.weight * output + (1 - self.weight) * sampled
        
        return result

if __name__ == '__main__':
    pass
