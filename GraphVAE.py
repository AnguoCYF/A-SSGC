
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class MLP_VAE(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, output_size: int) -> None:
        super(MLP_VAE, self).__init__()
        # 编码器
        self.encode_mu = nn.Linear(input_size, embedding_size)
        self.encode_logstd = nn.Linear(input_size, embedding_size)
        # 解码器
        self.decode = nn.Sequential(
            nn.Linear(embedding_size, output_size),
            nn.Sigmoid()
        )
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))

    def reparameterize(self, mu: Tensor, logstd: Tensor):
        if self.training:
            std = torch.exp(0.5 * logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, h: Tensor):
        mu = self.encode_mu(h)
        logstd = self.encode_logstd(h)
        z = self.reparameterize(mu, logstd)
        adj_recon = self.decode(z)  # 直接输出邻接矩阵概率
        return adj_recon, mu, logstd

class GraphVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        max_num_nodes: int,
    ) -> None:
        super(GraphVAE, self).__init__()
        self.max_num_nodes = max_num_nodes
        # 输入维度调整为节点特征展平后的维度
        self.vae = MLP_VAE(
            input_size=input_dim * max_num_nodes,
            embedding_size=latent_dim,
            output_size=max_num_nodes * (max_num_nodes + 1) // 2  # 上三角元素数
        )

    def recover_adj(self, l: Tensor):
        """从概率向量重构邻接矩阵"""
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes).to(device)
        triu_indices = torch.triu_indices(self.max_num_nodes, self.max_num_nodes).to(device)
        adj[triu_indices[0], triu_indices[1]] = l
        return adj + adj.t() - torch.diag(adj.diag())  # 对称化

    def forward(self, node_features: Tensor, adj_truth: Tensor):
        """
        Args:
            node_features: [batch_size, max_num_nodes, input_dim]
            adj_truth: [batch_size, max_num_nodes, max_num_nodes]
        """
        batch_size = node_features.size(0)
        # 展平节点特征作为输入
        # h_flat = node_features.reshape(batch_size, -1)
        h_flat = node_features.reshape(batch_size, -1) / math.sqrt(self.max_num_nodes)

        # VAE 生成邻接矩阵概率
        adj_recon_prob, mu, logstd = self.vae(h_flat)
        # 重构邻接矩阵
        adj_recon = torch.stack([self.recover_adj(prob) for prob in adj_recon_prob])
        # 计算重构损失（BCE）
        triu_mask = torch.triu(torch.ones_like(adj_truth, dtype=torch.bool), diagonal=0)
        loss_recon = F.binary_cross_entropy(
            adj_recon_prob,
            adj_truth[triu_mask].view(batch_size, -1)
        )
        # 计算 KL 散度
        loss_kl = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp()) / batch_size
        return loss_recon + loss_kl

    def generate(self, node_features: Tensor):
        """生成邻接矩阵"""
        with torch.no_grad():
            h_flat = node_features.reshape(1, -1)
            adj_recon_prob, _, _ = self.vae(h_flat)
            adj_recon = self.recover_adj(adj_recon_prob.squeeze())
            return (adj_recon > 0.5).float()  # 阈值化