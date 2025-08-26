import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from models.generate_adj import get_similarity_metrla, compute_support_gwn
from models.generate_adj import *
from models.fdf_backbone import (
    Diffusion,
    ST_decomposition,
    series_decomposition,
    ARIMAXModel,
    MultiLinearModel
)



class FDF(nn.Module):
    def __init__(self, args: Namespace):
        super(FDF, self).__init__()

        #self.decom = series_decomposition(kernel_size = 5)
        self.input_len = args.input_len
        self.device = args.device
        self.pred_len = args.pred_len
        self.time_steps = args.time_steps
        self.z_dim = args.z_dim
        self.num_clusters = args.num_clusters
        # 构建图结构
        adj_mat = get_similarity_metrla(thr=0.1)
        support = compute_support_gwn(adj_mat, device=self.device)
        node_num = adj_mat.shape[0]
        nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True).to(self.device)
        nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True).to(self.device)
        support.append([nodevec1, nodevec2])
        self.support = support


        self.decom = ST_decomposition(
            #channels=args.ST_channels,
            z_dim=args.z_dim,
            num_clusters=args.num_clusters,
            adj=self.support,
            device=self.device,
            seq_len=args.input_len,
            nheads=getattr(args, "nheads", 4),  # 如果 args 里没有 nheads，就用默认 4
            is_cross_t=getattr(args, "is_cross_t", False),  # 默认 False
            is_cross_s=getattr(args, "is_cross_s", False),  # 默认 False
            batch_size=args.batch_size,
            #tau=args.cluster_tau,
            #sep_margin=args.sep_margin,
        ).to(self.device)



        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            feature_dim=args.feature_dim,
            seq_len=args.input_len,
            pred_len=args.pred_len,
            MLP_hidden_dim=args.MLP_hidden_dim,
            emb_dim=args.emb_dim,
            adj=self.support,
            channels=args.ST_channels,
            z_dim=args.z_dim,
            device=self.device,
            beta_scheduler=args.scheduler,
            patch_size=args.patch_size
        )
        self.eta = 0
        
        self.seq_len = args.input_len

        # 趋势预测（ARIMAX）
        self.trend_linear = ARIMAXModel(
            seq_len=args.input_len,
            pred_len=args.pred_len,
            ar_order=1,
            diff_order=1,
            ma_order=1,
            exog_dim=1
        )
        #self.trend_linear = MultiLinearModel(seq_len = args.input_len, pred_len = args.pred_len)




    # 训练
    def pred(self, x):
        batch_size, input_len, num_features = x.size()
        
        x_seq = x[:, :self.seq_len, :]
        x_means = x_seq.mean(1, keepdim=True).detach()
        x_enc = x_seq - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        
        x_norm = x - x_means
        x_norm /= x_stdev
        
        x_seq_input = x_norm[:, :self.seq_len, :]

        #season_seq, trend_seq = self.decom(x_seq_input)

        season_seq, trend_seq, dist_seq, seq_loss,recon_loss, quant_loss, sparsity_loss, perplexity_loss = self.decom(x_seq_input)
        x_pred = x_norm[:, -self.pred_len:, :]
        #season_pred, trend_pred = self.decom(x_pred)

        season_pred, trend_pred, dist_pred, pred_loss,_recon_loss, _quant_loss, _sparsity_loss, _perplexity_loss = self.decom(x_pred)

        trend_seq = trend_seq.reshape(self.num_clusters, batch_size, self.input_len).permute(1, 2, 0)
        trend_pred = self.trend_linear(trend_seq)   #[B,pre_len,N_cluster]
        
        # Noising Diffusion
        t = torch.randint(0, self.time_steps, (batch_size,), device=self.device)

        season_flat = season_pred.reshape(batch_size, self.pred_len, -1)


        noise_season = self.diffusion(season_flat, t)
        season_pred = self.diffusion.pred(noise_season, t, season_flat)  #[B,pre_len,N, C]
        #融合

        assign_ids = dist_seq.argmax(axis=1).reshape(batch_size,-1)  #[B*N]
        # 扩展为[B, pred_len, N]（每个时间步共享相同的节点聚类分配）
        assign_ids = assign_ids.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, pred_len, N]
        # 增加维度用于gather操作：[B, pred_len, N, 1]
        #assign_ids_unsqueezed = assign_ids.unsqueeze(-1)
        #index = assign_ids_unsqueezed.expand(-1, -1, -1, self.z_dim)

        #trend_seq_reshaped = trend_seq.reshape(self.num_clusters, batch_size, self.input_len).permute(1, 2,0)  # [B, input_len, num_clusters]
        # 7. 根据聚类分配选择对应的趋势中心
        selected_centers = torch.gather(
            trend_pred,  # [B, pred_len, num_clusters, z_dim]
            dim=2,  # 在聚类维度上选择
            index=assign_ids  # [B, pred_len, N, z_dim]
        )  # [B, pred_len, N, z_dim]

        # 8. 融合残差与聚类中心（最终预测）
        selected_centers = selected_centers  # 移除z_dim维度：[B, pred_len, N]
        season_pred = season_pred.reshape(batch_size,self.pred_len,num_features,-1).mean(dim=-1)
        final_pred = season_pred + selected_centers  # 残差+对应聚类中心

        predict_x = final_pred
        
        dec_out = predict_x * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out,seq_loss,recon_loss, quant_loss, sparsity_loss, perplexity_loss

    # 验证或测试
    def forecast(self, input_x):
        x = input_x[:, :self.seq_len, :]
        b, _, dim = x.shape
        shape = torch.zeros((b, self.pred_len, dim), dtype=torch.int, device=self.device)
        
        x_means = x.mean(1, keepdim=True).detach()
        x_enc = x - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= x_stdev
        #season, trend = self.decom(x_enc)
        season, trend, q, loss,recon_loss, quant_loss, sparsity_loss, perplexity_loss = self.decom(x_enc)
        #trend_pred_part = trend
        trend_pred_part = trend.reshape(self.num_clusters,b, self.seq_len).permute(1, 2, 0)
        trend_pred = self.trend_linear(trend_pred_part)
        season_flat = season.reshape(b, self.pred_len, -1)

        #print(f"原始输入形状: {season_flat.shape}")  # 应为[B,L,N,z_dim]
        #print(f"原始输入形状: {season_flat.shape}")
        shape = torch.zeros((b, self.pred_len, dim * self.z_dim), dtype=torch.int, device=self.device)
        season_pred = self.diffusion.sample_infill(shape, self.time_steps, season_flat)

        assign_ids = q.argmax(axis=1).reshape(b, -1)  # [B*N]
        # 扩展为[B, pred_len, N]（每个时间步共享相同的节点聚类分配）
        assign_ids = assign_ids.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, pred_len, N]
        season_flat = season_flat.reshape(b, self.pred_len, dim, -1).mean(dim=-1)
        selected_centers = torch.gather(
            trend_pred,  # [B, pred_len, num_clusters, z_dim]
            dim=2,  # 在聚类维度上选择
            index=assign_ids  # [B, pred_len, N, z_dim]
        )

        selected_centers = selected_centers.reshape(b, self.pred_len, -1)
        #

        season_pred = season_pred.reshape(b, self.pred_len, dim, -1).mean(dim=-1)
        predict_x = selected_centers + season_pred
        #predict_x = trend_pred + season_pred
        dec_out = predict_x * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        x_pred = input_x[:, -self.pred_len:, :]
        #season_pred_part, trend_pred_part = self.decom(x_pred)
        return dec_out,loss,recon_loss, quant_loss, sparsity_loss, perplexity_loss

    def forward(self, x, task):
        if task == "train":
            return self.pred(x)  
        elif task == 'valid' or task == "test":
            return self.forecast(x)  
        else:
            raise ValueError(f"Invalid task: {task=}")
