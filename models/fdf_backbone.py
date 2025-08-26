import math
import torch
import torch.nn as nn
import numpy as np
from models.fdf_denoise_network import fdf_denoise_network
from functools import partial
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from models.layers import *
from einops import rearrange


def identity(x):
    return x

class moving_avg(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomposition(nn.Module):
    
    def __init__(self, kernel_size):
        super(series_decomposition, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class decomposition(nn.Module):

    def __init__(self, kernel_size):
        super(decomposition, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ST_decomposition0(nn.Module):
    def __init__(
        self,
        channels: int,
        z_dim: int,
        num_clusters: int,
        adj,
        device: torch.device,
        seq_len: int,
        nheads: int = 4,
        is_cross_t: bool = False,
        is_cross_s: bool = False,
        inputdim: int = 1,
        batch_size: int = 32,
        gmm_covariance_type: str = 'diag',
    ):
        super().__init__()
        self.adj = adj
        self.channels = channels
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        self.device = device
        self.seq_len = seq_len
        self.v = 1
        self.centers_initialized = False
        #self.centers = None  # To store the initialized cluster centers

        # 时空编码

        self.input_projection = Conv1d_with_init(inputdim, channels, 1)
        self.pre_time = TemporalLearning(channels, nheads, is_cross_t)
        self.pre_feature = FeatureLearning0(channels, nheads, is_cross_s)

        # GMM 软聚类
        self.gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type=gmm_covariance_type,
            init_params='kmeans',
            max_iter=50,
            warm_start=True,
            random_state=0,
        )
        self.register_buffer('centers', None)

    def initialize_centers(self, x: torch.Tensor):
        """Initialize cluster centers using GMM on the first batch of data"""
        if self.centers_initialized:
            return

        B, L, N = x.shape
        # 1) ST 编码
        x_enc = self.input_projection(
            x.unsqueeze(-1).permute(0, 3, 2, 1).reshape(B, 1, N * L)
        ).reshape(B, self.channels, N, L)
        base_shape = x_enc.shape
        t_feat = self.pre_time(x_enc, base_shape)
        s_feat = self.pre_feature(t_feat, base_shape, self.adj).reshape(B, self.channels, N, L)

        # 2) 提取特征并归一化
        feats = s_feat.permute(0, 2, 1, 3).reshape(B * N, -1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        Z = F.normalize(feats, p=2, dim=1, eps=1e-8)

        # 3) GMM initialization only
        Z_cpu = Z.detach().cpu().numpy()
        self.gmm.fit(Z_cpu)

        # Store the initialized centers
        with torch.no_grad():
            self.centers = torch.from_numpy(self.gmm.means_.astype(np.float32)).to(Z.device)

        self._centers_initialized = True

        # No longer need the GMM after initialization


    def forward(self, x: torch.Tensor):

        B, L, N = x.shape
        # 1) ST 编码
        x_enc = self.input_projection(
            x.unsqueeze(-1).permute(0, 3, 2, 1).reshape(B, 1, N * L)
        ).reshape(B, self.channels, N, L)
        base_shape = x_enc.shape
        t_feat = self.pre_time(x_enc, base_shape)
        s_feat = self.pre_feature(t_feat, base_shape, self.adj).reshape(B, self.channels, N, L)

        # 2) 提取特征并归一化
        feats = s_feat.permute(0, 2, 1, 3).reshape(B * N, -1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        Z = F.normalize(feats, p=2, dim=1, eps=1e-8)

        # 3) GMM 训练 & 软分配
        Z_cpu = Z.detach().cpu().numpy()
        if self.training:
            self.gmm.fit(Z_cpu)

        # 获取聚类中心（由GMM初始化）
        centers = torch.from_numpy(self.gmm.means_.astype(np.float32)).to(Z.device)

        # 计算距离平方
        distances_sq = torch.cdist(Z, centers).pow(2)  # 形状: [B*N, num_clusters]
        cluster_ids = torch.argmin(distances_sq, dim=1)

        # 基于距离度量计算Qq
        q = 1.0 / (1.0 + distances_sq / self.v)

        # 对q进行处理：幂运算后归一化
        q = q.pow((self.v + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)  # 沿聚类维度求和归一化

        # 4) 确定最近的聚类中心
        #dists = torch.cdist(Z, centers)  # L2距离
        #cluster_ids = torch.argmin(dists, dim=1)

        #cluster_ids = torch.argmin(q, dim=1) #q.argmax(axis=1)

        counts = torch.bincount(cluster_ids)
        print(counts)

        # 5) 计算残差（residual -> seasonality）
        res = (Z - centers[cluster_ids]) \
            .view(B, N, L, self.channels) \
            .permute(0, 2, 1, 3)


        return res, centers, distances_sq, Z  ##[B,L,N,C]  [n_cluster,B*L] q[[B*N, num_clusters]] B, L, N

    def visualize_clusters(self, Z, centers, cluster_ids):
        # 将特征和聚类中心拼接在一起
        data_plot = Z.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()
        data_plot = np.concatenate((data_plot, centers_np), axis=0)

        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne_2d = tsne.fit_transform(data_plot)

        # 分离数据点和聚类中心
        data_tsne = data_tsne_2d[:-self.num_clusters, :]
        center = data_tsne_2d[-self.num_clusters:, :]

        # 绘制散点图
        cluster_ids_np = cluster_ids.detach().cpu().numpy()
        data_dict = {
            str(label): np.array([data_tsne[idx] for idx, cluster in enumerate(cluster_ids_np) if cluster == label])
            for label in range(self.num_clusters)
        }
        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(12, 10))
        for m, data0 in data_dict.items():
            if data0.size > 0:
                plt.scatter(data0[:, 0], data0[:, 1], color=cmap(int(m)), label=f'label.{m}: {len(data0)}', s=30,
                            alpha=0.5)
                plt.scatter(center[int(m), 0], center[int(m), 1], color=cmap(int(m)), marker='x', s=30, alpha=0.5)

        plt.legend(fontsize=22)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig("./results/tsne_clusters.png")
        plt.close()


class ST_decomposition00(nn.Module):
    def __init__(
        self,
        #channels: int,
        z_dim: int,
        num_clusters: int,
        adj,
        device: torch.device,
        seq_len: int,
        nheads: int = 4,
        is_cross_t: bool = False,
        is_cross_s: bool = False,
        inputdim: int = 1,
        batch_size: int = 4,
        commitment_cost: float = 0.25,
        dropout: float = 0.2,
        alpha: float = 0.2
    ):

        super().__init__()
        self.adj = adj
        #self.channels = channels
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Initial projection
        self.input_projection = Conv1d_with_init(inputdim, z_dim, 1)

        # Temporal and spatial feature learning modules
        self.pre_time = TemporalLearning(z_dim, nheads, is_cross_t)
        self.pre_feature = FeatureLearning0(z_dim, nheads, is_cross_s)


        # Bottleneck layer to z_dim
        #self.bottleneck = nn.Conv2d(channels, z_dim, kernel_size=1)

        # Spatial attention for soft clustering
        self.spatial_attention = SpatialAttention(z_dim)

        # Graph attention for spatial dependencies
        self.gat = nn.ModuleList([
            GraphAttentionLayer(z_dim, z_dim, dropout=dropout, alpha=alpha)
            for _ in range(2)
        ])

        # 层次化量化器 - 码本维度为 [n_cluster, B*L]
        self.vq_primary_spatial = SpatialVectorQuantizer(num_clusters, batch_size, seq_len, commitment_cost)
        self.vq_secondary_spatial = SpatialVectorQuantizer(num_clusters, batch_size, seq_len, commitment_cost)

        # 注册码本缓冲区 - 现在是正确的维度
        self.register_buffer('codebook_primary', torch.zeros(num_clusters, batch_size * seq_len))
        self.register_buffer('codebook_secondary', torch.zeros(num_clusters, batch_size * seq_len))

        # Soft clustering layer
        self.soft_cluster_layer = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, num_clusters),
            nn.Softmax(dim=-1)
        )

        # Decoders
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, kernel_size=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(),
            nn.Conv2d(z_dim, z_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU()
        )

        # Residual decoder
        self.res_decoder = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, kernel_size=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU()
        )

        # Output projection
        self.output_projection = nn.Conv2d(z_dim, inputdim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        B, L, N = x.shape

        # 1. ST encoding
        x_enc = self.input_projection(
            x.unsqueeze(-1).permute(0, 3, 2, 1).reshape(B, 1, N * L)
        ).reshape(B, self.z_dim, N, L)
        base_shape = x_enc.shape

        # 2. Temporal and spatial feature learning
        t_feat = self.pre_time(x_enc, base_shape)
        s_feat = self.pre_feature(t_feat, base_shape, self.adj)  #[B, C N, L]

        # 3. Dimensionality reduction
        #z = s_feat #self.bottleneck(s_feat)  # [B, z_dim, N, L]

        # 4. Apply spatial attention for soft clustering
        z_attended = self.spatial_attention(s_feat)  # [B, z_dim, N, L]

        # 5. 空间维度量化 - 将通道和空间维度一起压缩
        # [B, z_dim, N, L] -> [B, L, z_dim*N]
        z_reshaped = z_attended.permute(0, 3, 1, 2).reshape(B, L, -1)  # [B, L, z_dim*N]


        # 第一级量化
        primary_loss, primary_quantized, primary_perplexity, primary_encodings = self.vq_primary_spatial(z_reshaped)

        ## 计算残差
        #primary_residual = z_reshaped - primary_quantized

        # 第二级量化
        secondary_loss, secondary_quantized, secondary_perplexity, secondary_encodings = self.vq_secondary_spatial(
            primary_quantized)

        # 合并两级量化结果
        quantized = primary_quantized + secondary_quantized  # [B, L, z_dim*N]

        # 计算总量化损失
        quant_loss = primary_loss + secondary_loss

        # 将量化结果转回原始维度 [B, L, z_dim*N] -> [B, z_dim, N, L]
        quantized_reshaped = quantized.reshape(B, L, self.z_dim, N).permute(0, 2, 3, 1)  # [B, z_dim, N, L]

        # 保存码本
        if self.training:
            with torch.no_grad():
                self.codebook_primary = self.vq_primary_spatial.embedding.weight.data
                self.codebook_secondary = self.vq_secondary_spatial.embedding.weight.data

        # 6. Apply GAT for spatial dependencies
        z_graph = z_attended.permute(0, 3, 2, 1)  # [B, L, N, z_dim]
        gat_outputs = []

        for b in range(B):
            batch_outputs = []
            for t in range(L):
                node_feats = z_graph[b, t]  # [N, z_dim]
                # Apply GAT layers
                x_gat = F.dropout(node_feats, 0.2, training=self.training)
                for gat_layer in self.gat:
                    x_gat = gat_layer(x_gat, self.adj)
                batch_outputs.append(x_gat.unsqueeze(0))  # [1, N, z_dim]

            batch_outputs = torch.cat(batch_outputs, dim=0).unsqueeze(0)  # [1, L, N, z_dim]
            gat_outputs.append(batch_outputs)

        gat_outputs = torch.cat(gat_outputs, dim=0)  # [B, L, N, z_dim]

        # 7. Compute soft cluster assignments
        gat_flat = gat_outputs.reshape(-1, self.z_dim)  # [B*L*N, z_dim]
        soft_assignments = self.soft_cluster_layer(gat_flat)  # [B*L*N, num_clusters]
        soft_assignments = soft_assignments.reshape(B, L, N, self.num_clusters)  # [B, L, N, num_clusters]

        # 计算与二级码本中心的残差
        z_attended_flat = z_attended.permute(0, 3, 1, 2).reshape(B, L, -1)  # [B, L, z_dim*N]
        secondary_residuals = []

        #z_attended_reshaped = s_feat.permute(0, 3, 1, 2).reshape(B, L, -1) #[B, C N, L]

        for k in range(self.num_clusters):
            # 获取第k个码本条目 [B*L]，需要重塑以匹配空间特征
            center_pattern = self.codebook_secondary[k].reshape(B, L)  # [B, L]
            # 为每个空间特征维度复制这个模式
            center_expanded = center_pattern.unsqueeze(-1).repeat(1, 1, self.z_dim * N)  # [B, L, z_dim*N]
            res = z_attended_flat - center_expanded  # [B, L, z_dim*N]
            secondary_residuals.append(res.unsqueeze(3))  # [B, L, z_dim*N, 1]

        secondary_residuals = torch.cat(secondary_residuals, dim=3)  # [B, L, z_dim*N, num_clusters]

        # 重新组织维度为 [B, L, N, num_clusters, z_dim]
        secondary_residuals = secondary_residuals.reshape(B, L, self.z_dim, N, self.num_clusters)
        secondary_residuals = secondary_residuals.permute(0, 1, 3, 4, 2)  # [B, L, N, num_clusters, z_dim]

        # 9. Weight residuals by soft cluster assignments
        soft_assignments_expanded = soft_assignments.unsqueeze(-1)  # [B, L, N, num_clusters, 1]
        weighted_secondary_res = secondary_residuals * soft_assignments_expanded  # [B, L, N, num_clusters, z_dim]
        weighted_secondary_res = weighted_secondary_res.sum(dim=3)  # [B, L, N, z_dim]

        # 10. Reconstruction
        decoded_quant = self.decoder(quantized_reshaped)  # [B, channels, N, L]
        residual_features = weighted_secondary_res.permute(0, 3, 2, 1)  # [B, z_dim, N, L]
        decoded_res = self.res_decoder(residual_features)  # [B, channels, N, L]
        decoded_combined = decoded_quant + decoded_res  # [B, channels, N, L]

        # 输出投影
        output = self.output_projection(decoded_combined)  # [B, inputdim, N, L]
        output = output.permute(0, 3, 2, 1).squeeze(-1)  # [B, L, N]

        # 计算损失
        recon_loss = F.mse_loss(output, x)
        sparsity_loss = torch.mean(torch.abs(weighted_secondary_res))
        perplexity_loss = -0.1 * (torch.log(primary_perplexity + 1e-10) + torch.log(secondary_perplexity + 1e-10))

        total_loss = recon_loss + quant_loss + 0.1 * sparsity_loss + perplexity_loss
        '''
        return {
            'output': output,
            'z': z_attended,
            'quantized': quantized_reshaped,
            'secondary_residual': weighted_secondary_res,
            'soft_assignments': soft_assignments,
            'primary_quantized': primary_quantized,
            'secondary_quantized': secondary_quantized,
            'primary_encodings': primary_encodings,
            'secondary_encodings': secondary_encodings,
            'recon_loss': recon_loss,
            'quant_loss': quant_loss,
            'sparsity_loss': sparsity_loss,
            'perplexity_loss': perplexity_loss,
            'total_loss': total_loss,
            'primary_perplexity': primary_perplexity,
            'secondary_perplexity': secondary_perplexity
        }'''


        res = weighted_secondary_res  # [B, L, N, z_dim]
        centers = self.codebook_secondary.reshape(self.num_clusters,B*L)
        dists = torch.cdist(s_feat.permute(1, 2, 0, 3).reshape(-1, B*L), centers)  # L2距离
        # cluster_ids = torch.argmin(dists, dim=1)
        #[]feats = s_feat.permute(0, 2, 1, 3).reshape(B * N, -1)   s_feat [B, C N, L]  [C*N,B*L] [n_cluster,B*L]  [C*N, num_clusters]

        return res, centers, dists, total_loss,  recon_loss, quant_loss, sparsity_loss, perplexity_loss ##[B,L,N,C]  [n_cluster,B*L] q[[C*N, num_clusters]] B, L, N


class ST_decomposition(nn.Module):
    def __init__(
            self,
            z_dim: int,
            num_clusters: int,
            adj,
            device: torch.device,
            seq_len: int,
            nheads: int = 4,
            is_cross_t: bool = False,
            is_cross_s: bool = False,
            inputdim: int = 1,
            batch_size: int = 4,
            commitment_cost: float = 0.25,
            dropout: float = 0.5,
            alpha: float = 0.2
    ):
        super().__init__()
        self.adj = adj
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.commitment_cost = commitment_cost
        self.dropout = dropout

        # Modules
        self.input_projection = Conv1d_with_init(inputdim, z_dim, 1)
        self.pre_time = TemporalLearning(z_dim, nheads, is_cross_t)
        self.pre_feature = FeatureLearning0(z_dim, nheads, is_cross_s)
        self.spatial_attention = SpatialAttention(z_dim)
        self.gat = nn.ModuleList([
            GraphAttentionLayer(z_dim, z_dim, dropout=self.dropout, alpha=alpha) for _ in range(2)
        ])
        self.soft_cluster_layer = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(z_dim, num_clusters), nn.Softmax(dim=-1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, kernel_size=1), nn.BatchNorm2d(z_dim), nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(z_dim, z_dim, kernel_size=3, padding=1), nn.BatchNorm2d(z_dim), nn.ReLU()
        )
        self.res_decoder = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, kernel_size=1), nn.BatchNorm2d(z_dim), nn.ReLU()
        )
        self.output_projection = nn.Conv2d(z_dim, inputdim, kernel_size=1)

        # 量化器将在第一次前向传播时初始化
        self.vq_primary = None
        self.vq_secondary = None

    def forward(self, x: torch.Tensor):
        B, L, N = x.shape

        # 在第一次运行时初始化量化器 - 注意这里使用B*L作为embedding_dim
        if self.vq_primary is None:
            self.vq_primary = SpatialVectorQuantizer(
                self.num_clusters, B, L, self.commitment_cost
            ).to(self.device)
            self.vq_secondary = SpatialVectorQuantizer(
                self.num_clusters, B, L, self.commitment_cost
            ).to(self.device)

        # 1. 编码
        x_enc = self.input_projection(
            x.unsqueeze(-1).permute(0, 3, 2, 1).reshape(B, 1, N * L)
        ).reshape(B, self.z_dim, N, L)
        base_shape = x_enc.shape
        t_feat = self.pre_time(x_enc, base_shape)
        s_feat = self.pre_feature(t_feat, base_shape, self.adj)  #
        z_attended = self.spatial_attention(s_feat)

        # 2. 层次化量化
        z_reshaped = z_attended.permute(0, 3, 1, 2).reshape(B, L, -1)  # [B, L, z_dim*N]

        # 第一级量化
        primary_loss, primary_quantized, primary_perplexity, primary_indices = self.vq_primary(z_reshaped)

        # 第二级量化对残差进行
        primary_residual = z_reshaped - primary_quantized
        secondary_loss, secondary_quantized, secondary_perplexity, secondary_indices = self.vq_secondary(
            primary_residual)

        quant_loss = primary_loss + secondary_loss
        quantized = primary_quantized + secondary_quantized
        quantized_reshaped = quantized.reshape(B, L, self.z_dim, N).permute(0, 2, 3, 1)  # [B, z_dim, N, L]

        # 3. GAT和软聚类
        z_graph = z_attended.permute(0, 3, 2, 1)
        gat_outputs = []

        for b in range(B):
            batch_outputs = []
            for t in range(L):
                node_feats = z_graph[b, t]  # [N, z_dim]
                x_gat = F.dropout(node_feats, self.dropout, training=self.training)
                for gat_layer in self.gat:
                    x_gat = gat_layer(x_gat, self.adj)
                batch_outputs.append(x_gat.unsqueeze(0))

            batch_outputs = torch.cat(batch_outputs, dim=0).unsqueeze(0)
            gat_outputs.append(batch_outputs)

        gat_outputs = torch.cat(gat_outputs, dim=0)  # [B, L, N, z_dim]
        soft_assignments = self.soft_cluster_layer(gat_outputs.view(-1, self.z_dim)).view(B, L, N, self.num_clusters)

        # 4. 残差计算和加权 - 现在二级码本是 [n_cluster, B*L]
        secondary_codebook = self.vq_secondary.embedding.weight.data  # [n_cluster, B*L]
        z_attended_flat = z_attended.permute(0, 3, 1, 2).reshape(B, L, -1)  # [B, L, z_dim*N]

        # 计算与每个码本向量的残差
        secondary_residuals = []
        for k in range(self.num_clusters):
            # 获取第k个码本条目 [B*L]，重塑为 [B, L]
            center_pattern = secondary_codebook[k].reshape(B, L)  # [B, L]
            # 为每个空间特征维度复制这个模式
            center_expanded = center_pattern.unsqueeze(-1).repeat(1, 1, self.z_dim * N)  # [B, L, z_dim*N]
            res = z_attended_flat - center_expanded  # [B, L, z_dim*N]
            secondary_residuals.append(res.unsqueeze(3))  # [B, L, z_dim*N, 1]

        secondary_residuals = torch.cat(secondary_residuals, dim=3)  # [B, L, z_dim*N, num_clusters]

        # 重新组织维度为 [B, L, N, num_clusters, z_dim]
        secondary_residuals = secondary_residuals.reshape(B, L, self.z_dim, N, self.num_clusters)
        secondary_residuals = secondary_residuals.permute(0, 1, 3, 4, 2)  # [B, L, N, num_clusters, z_dim]

        # 通过软分配加权残差
        weighted_res = (secondary_residuals * soft_assignments.unsqueeze(-1)).sum(dim=3)  # [B, L, N, z_dim]

        # 5. 重构
        decoded_quant = self.decoder(quantized_reshaped)
        decoded_res = self.res_decoder(weighted_res.permute(0, 3, 2, 1))
        decoded_combined = decoded_quant + decoded_res
        output = self.output_projection(decoded_combined).permute(0, 3, 2, 1).squeeze(-1)

        # 6. 损失计算
        recon_loss = F.mse_loss(output, x)
        sparsity_loss = torch.mean(torch.abs(weighted_res))

        # 困惑度惩罚 - 鼓励使用更多的码本条目
        perplexity_penalty = -(primary_perplexity + secondary_perplexity)

        # 调整损失权重以防止过拟合
        total_loss = recon_loss + 0.5 * quant_loss + 0.05 * sparsity_loss + 0.01 * perplexity_penalty

        # 返回结果
        res = weighted_res  # [B, L, N, z_dim]
        centers = secondary_codebook  # [n_cluster, B*L] - 现在是正确的维度

        # 计算距离 - 现在s_feat需要重塑以匹配centers的维度
        s_feat_reshaped = s_feat.permute(1, 2, 0, 3).reshape(-1, B * L)  # [z_dim*N, B*L]
        dists = torch.cdist(s_feat_reshaped, centers)  # [z_dim*N, n_cluster]

        return res, centers, dists, total_loss, recon_loss, quant_loss, sparsity_loss, perplexity_penalty
    #



class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        feature_dim : int,
        seq_len : int,
        pred_len : int,
        MLP_hidden_dim : int,
        emb_dim : int,
        patch_size : int,
        adj: None,
        channels: int,
        z_dim: int,
        device: torch.device,
        beta_scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps
        self.seq_length = seq_len
        self.pred_length = pred_len
        self.adj = adj

        if beta_scheduler == 'cosine':
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif beta_scheduler == 'linear':
            self.betas = self._linear_beta_schedule().to(self.device)
        elif beta_scheduler == 'exponential':
            self.betas = self._exponential_beta_schedule().to(self.device)
        elif beta_scheduler == 'inverse_sqrt':
            self.betas = self._inverse_sqrt_beta_schedule().to(self.device)
        elif beta_scheduler == 'piecewise':
            self.betas = self._piecewise_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Unknown schedule type: {scheduler}")
        
        self.eta = 0
        self.alpha = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.denoise_net = fdf_denoise_network(feature_dim, seq_len, pred_len, device, self.adj, channels,
                                               z_dim, MLP_hidden_dim, emb_dim, patch_size)

        #self.denoise_net = fdf_denoise_network(feature_dim, seq_len, pred_len, device, MLP_hidden_dim, emb_dim, patch_size)
    
    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alpha_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def _exponential_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        steps = self.time_steps
        return beta_start * ((beta_end / beta_start) ** (torch.linspace(0, 1, steps)))

    def _inverse_sqrt_beta_schedule(self, beta_start=1e-4):
        steps = self.time_steps
        x = torch.arange(1, steps + 1)
        return torch.clip(beta_start / torch.sqrt(x), 0, 0.999)

    def _piecewise_beta_schedule(self, beta_values=[1e-4, 0.01, 0.02], segment_steps=[100, 200, 300]):
        assert len(beta_values) == len(segment_steps), "beta_values and segment_steps length mismatch"
        betas = [torch.full((steps,), beta) for beta, steps in zip(beta_values, segment_steps)]
        return torch.cat(betas)[:self.time_steps]
    
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)  
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1).unsqueeze(-1)  
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x, t):
        noisy_x, _ = self.noise(x, t)
        return noisy_x
    
    def pred(self, x, t, cond):
        if t == None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=self.device)
        return self.denoise_net(x, t, cond)
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, t, cond, clip_x_start=False, padding_masks=None):

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.denoise_net(x, t, cond)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
        
    @torch.no_grad()
    def sample_infill(self, shape, sampling_timesteps, cond, clip_denoised=True):
        batch_size, _, _ = shape.shape
        batch, device, total_timesteps, eta = shape[0], self.device, self.time_steps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  
        shape = shape
        denoise_series = torch.randn(shape.shape, device=device)   #[B,pred_len,N]

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(denoise_series, time_cond, cond, clip_x_start=clip_denoised)
            #denoise_series[B,L,N] time_cond [4] time 49 cond [4,96,828]
            if time_next < 0:
                denoise_series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(denoise_series)

            denoise_series = pred_mean + sigma * noise

        return denoise_series

class ARIMAXModel(nn.Module):
    def __init__(self, seq_len, pred_len, ar_order=1, diff_order=1, ma_order=1, exog_dim=0):

        """
                ARIMAX模型（带外部特征的ARIMA）

                Args:
                    seq_len: 输入序列长度
                    pred_len: 预测序列长度
                    ar_order: 自回归阶数 (p)
                    diff_order: 差分阶数 (d)
                    ma_order: 移动平均阶数 (q)
                    exog_dim: 外部特征维度 (0表示无外部特征)
                """
        super(ARIMAXModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ar_order = ar_order
        self.diff_order = diff_order
        self.ma_order = ma_order
        self.exog_dim = exog_dim
        # ARIMA核心组件
        self.ar_weights = nn.Parameter(torch.randn(ar_order)* 0.1)
        self.ma_weights = nn.Parameter(torch.randn(ma_order)* 0.1)

        # 外部特征处理层
        if exog_dim > 0:
            self.exog_projection = nn.Sequential(
                nn.Linear(exog_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1))

        # 差分后序列长度
        self.diff_seq_len = seq_len - diff_order

        # 最终投影层（处理组合特征）
        self.linear_projection = nn.Linear(self.diff_seq_len, pred_len)

        # 残差缓冲区（用于MA计算）
        self.register_buffer('residuals', torch.zeros(ma_order, dtype=torch.float32))

    def diff(self, x, order):
        """
        Differencing operation
        """
        for _ in range(order):
            x = x[:, 1:] - x[:, :-1]
        return x

    def forward(self, x, exog=None):
        """
                输入：
                    x: 主序列 [batch, seq_len, 1]
                    exog: 外部特征 [batch, seq_len, exog_dim] (可选)
                输出：
                    [batch, pred_len, 1]
                """

        batch_size, seq_len, feature_dim = x.shape
        # 差分处理
        x_diff = self.diff(x, self.diff_order) if self.diff_order > 0 else x

        # 自回归项（修复切片错误）
        ar_terms = torch.zeros_like(x_diff)
        for i in range(self.ar_order):
            slice_end = -i if i > 0 else None
            ar_terms[:, i:] += self.ar_weights[i] * x_diff[:, :slice_end]

            # 移动平均项（使用历史残差）
        ma_terms = torch.zeros_like(x_diff)
        if self.ma_order > 0:
            current_residual = (x_diff - ar_terms)[:, -1, 0]  # 最新残差
            self._update_residuals(current_residual)
            for i in range(self.ma_order):
                ma_terms[:, i:] += self.ma_weights[i] * self.residuals[i]



        # 外部特征处理
        #exog_effect = 0
        if self.exog_dim > 0 and exog is not None:
            exog_effect = self.exog_projection(exog).squeeze(-1)
            exog_effect = exog_effect[:, self.diff_order:].unsqueeze(-1)  # 对齐差分后长度
        else:
            exog_effect = torch.zeros_like(x_diff)  # 保持张量形式
        # 组合所有成分
        combined = x_diff + ar_terms + ma_terms + exog_effect

        # 维度调整并投影
        output = self.linear_projection(combined.permute(0, 2, 1))
        return output.permute(0, 2, 1)

    def _update_residuals(self, batch_residuals):
        """处理batch维度的残差更新"""
        if self.ma_order == 0:
            return

        # 方法1：取batch平均
        mean_residual = batch_residuals.mean().detach()

        # 方法2：取最后一个样本（任选一种）
        # mean_residual = batch_residuals[-1].detach()

        # 安全更新
        with torch.no_grad():
            new_buffer = torch.roll(self.residuals, 1)
            new_buffer[0] = mean_residual
            self.residuals.copy_(new_buffer)

    def reset_residuals(self):
        """重置残差缓冲区"""
        if self.ma_order > 0:
            self.residuals.zero_()


class MultiLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len, num_loops=2):
        super(MultiLinearModel, self).__init__()

        self.linear_projection = nn.Linear(seq_len, pred_len, bias=True)
        self.weighted_linear = nn.Linear(num_loops, 1, bias=True)
        self.num_loops = num_loops

    def forward(self, input_data):
        transformed_data = [input_data.unsqueeze(-1)]   #[B,seq_len,C] --->[B,seq_len,C,1]
        
        for i in range(2, self.num_loops + 1):
            transformed = input_data.clone()
            xxx=input_data[:, 1, :]  #[B,C]
            transformed[:, 1, :] = torch.sign(input_data[:, 1, :]) * (torch.abs(input_data[:, 1, :]) ** (1 / i))   #平方根变换	压缩幅度，增强小值特征，抑制大值特征
            transformed_data.append(transformed.unsqueeze(-1))
        
        concatenated_data = torch.cat(transformed_data, dim=-1)
        sequence_output = self.linear_projection(concatenated_data.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        output = self.weighted_linear(sequence_output).squeeze(-1)
        #[B,Pre_len,C]
        return output
