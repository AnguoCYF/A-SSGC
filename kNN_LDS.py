import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from sklearn.metrics import precision_score, recall_score, f1_score
from dgl.nn import GraphConv, GATConv, GINConv
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def generate_masks(features, labels, train_ratio, valid_ratio=0.2, seed=42, resample=False):
    if resample:
        sm = SMOTE(random_state=seed)
        features, labels = sm.fit_resample(features, labels)

    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 第一次分割：训练集 + 测试集
    _, _, _, _, train_idx, test_idx = train_test_split(
        features, labels, np.arange(num_nodes),
        train_size=train_ratio,
        random_state=seed
    )

    # 第二次分割：从训练集中划分出验证集
    train_sub_idx, valid_idx = train_test_split(
        train_idx,
        train_size=1 - valid_ratio,  # 训练集占原始训练集的比例
        random_state=seed
    )

    # 生成掩码
    train_mask[train_sub_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True

    return features, labels, train_mask, valid_mask, test_mask


class LearnableGraphStructure(nn.Module):
    """可学习的图结构生成模块"""

    def __init__(self, num_nodes, init_method='knn', k=5, features=None):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # kNN初始化
        if init_method == 'knn' and features is not None:
            with torch.no_grad():
                knn_adj = kneighbors_graph(features, k, metric='cosine').toarray()
                self.theta.data = torch.tensor(knn_adj, dtype=torch.float32)

        # 上三角稀疏约束
        mask = torch.triu(torch.ones_like(self.theta), diagonal=1)
        self.register_buffer('mask', mask)

    def sample_adj(self, training=True):
        prob = torch.sigmoid(self.theta) * self.mask

        if training:
            # Gumbel-Softmax风格重参数化
            noise = torch.rand_like(prob)
            sample = (noise < prob).float()
            adj = sample + (prob - prob.detach())  # 直通估计器
        else:
            adj = (prob > 0.5).float() + (prob - prob.detach())  # 直通估计器

        return adj


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, out_dim, allow_zero_in_degree=True)

    def forward(self, g, features):
        # 传递 edge_weight 到卷积层
        h = F.relu(self.conv1(g, features, edge_weight=g.edata['w']))
        h = self.conv2(g, h, edge_weight=g.edata['w'])
        return h

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads, activation=F.relu, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, num_heads=1, activation=None, allow_zero_in_degree=True)

    def forward(self, g, features):
        h = F.relu(self.conv1(g, features, edge_weight=g.edata['w'])).flatten(1)
        h = self.conv2(g, h, edge_weight=g.edata['w']).squeeze(1)
        return h

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(GIN, self).__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ), aggregator_type='sum')
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        h = self.embedding(features)
        for conv in self.layers:
            h = conv(g, h, edge_weight=g.edata['w'])
            h = F.relu(h)
        h = self.readout(h)
        return h

class LDS_Trainer:
    """LDS-GNN训练框架（严格遵循论文算法）"""

    def __init__(self, features, labels, train_mask, val_mask, test_mask, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化图结构学习器
        self.graph_learner = LearnableGraphStructure(
            features.shape[0],
            init_method=config['graph_init'],
            k=config['k'],
            features=features
        ).to(self.device)

        # 初始化GCN
        self.model = GCN(
            in_dim=features.shape[1],
            hidden_dim=config['hidden_dim'],
            out_dim=len(np.unique(labels))
        ).to(self.device)

        # 根据 config 选择图分类器
        if config['model_type'] == 'GCN':
            self.model = GCN(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                out_dim=len(np.unique(labels))
            ).to(self.device)
        elif config['model_type'] == 'GIN':
            self.model = GIN(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                out_dim=len(np.unique(labels)),
                num_layers=config.get('num_layers', 2)
            ).to(self.device)
        elif config['model_type'] == 'GAT':
            self.model = GAT(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                out_dim=len(np.unique(labels)),
                num_heads=config.get('num_heads', 4)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {config['model_type']}")


        # 优化器（分离参数）
        self.optim_model = torch.optim.Adam(
            self.model.parameters(),
            lr=config['inner_lr'],
            weight_decay=config['l2_reg']
        )
        self.optim_graph = torch.optim.SGD(
            [self.graph_learner.theta],
            lr=config['outer_lr']
        )

        # 数据准备
        self.features = torch.FloatTensor(features).to(self.device)
        self.labels = torch.LongTensor(labels).to(self.device)
        self.train_mask = torch.tensor(train_mask).to(self.device)
        self.val_mask = torch.tensor(val_mask).to(self.device)
        self.test_mask = torch.tensor(test_mask).to(self.device)

    def train_step(self):
        """内层优化：固定θ优化模型参数"""
        self.model.train()

        # 采样邻接矩阵
        adj = self.graph_learner.sample_adj(training=False)

        # 构建DGL图
        # g = dgl.from_scipy(sp.csr_matrix(adj.cpu().detach().numpy()))
        rows, cols = torch.where(adj > 0)
        edge_weights = adj[rows, cols]
        g = dgl.graph((rows, cols), num_nodes=adj.shape[0]).to(self.device)
        g.edata['w'] = edge_weights
        g = g.to(self.device)

        # 前向传播
        logits = self.model(g, self.features)
        loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])

        # 反向传播
        self.optim_model.zero_grad()
        loss.backward()
        self.optim_model.step()
        return loss.item()

    def update_graph_structure(self, val_loss):
        """外层优化：固定模型优化θ"""
        self.optim_graph.zero_grad()
        val_loss.backward()

        # 梯度投影（仅保留上三角部分）
        with torch.no_grad():
            if self.graph_learner.theta.grad is None:
                raise RuntimeError("theta.grad 未计算！检查计算图连接性。")

            # 应用稀疏性约束
            grad = self.graph_learner.theta.grad * self.graph_learner.mask
            self.graph_learner.theta.data -= self.config['outer_lr'] * grad

        return val_loss.item()

    def evaluate(self, mask):
        """评估函数 (支持任意掩码)"""
        self.model.eval()
        with torch.set_grad_enabled(True):  # 评估时不追踪梯度
            adj = self.graph_learner.sample_adj(training=False)
            rows, cols = torch.where(adj > 0)  # 获取非零边的行列索引
            edge_weights = adj[rows, cols]  # 边的概率值
            g = dgl.graph((rows, cols), num_nodes=adj.shape[0]).to(self.device)
            g.edata['w'] = edge_weights

            logits = self.model(g, self.features)
            loss = F.cross_entropy(logits[mask], self.labels[mask])

            # 转换为numpy计算指标
            preds = logits[mask].argmax(1).cpu().numpy()
            labels = self.labels[mask].cpu().numpy()

            acc = (preds == labels).mean()
            prec = precision_score(labels, preds, average='macro', zero_division=0)
            recall = recall_score(labels, preds, average='macro', zero_division=0)
            f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return loss, acc, prec, recall, f1

    def train(self):
        """双层优化训练循环"""
        best_val_acc = 0
        patience = 0

        for epoch in range(self.config['max_epochs']):
            # 内层优化（多次参数更新）
            for _ in range(self.config['inner_steps']):
                train_loss = self.train_step()

            # 外层优化（基于验证损失）
            val_loss, val_acc, _, _, _ = self.evaluate(self.val_mask)
            _, test_acc, test_prec, test_recall, test_f1 = self.evaluate(self.test_mask)

            self.update_graph_structure(val_loss)

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Acc: {test_acc:.3f}\n"
                  f"Test Metrics: Prec {test_prec:.3f}, Recall {test_recall:.3f}, F1 {test_f1:.3f}")

            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                # 保存最佳模型时的测试指标
                best_test_metrics = (test_acc, test_prec, test_recall, test_f1)
            else:
                patience += 1
                if patience >= self.config['patience']:
                    break

        print(f"\nBest Test Metrics: "
              f"Acc: {best_test_metrics[0]:.3f}, "
              f"Prec: {best_test_metrics[1]:.3f}, "
              f"Recall: {best_test_metrics[2]:.3f}, "
              f"F1: {best_test_metrics[3]:.3f}")


if __name__ == "__main__":
    # %% Data Format for glass Dataset
    df = pd.read_csv('glass.csv')
    features = df.drop(columns=['Type']).values
    labels, _ = pd.factorize(df['Type'])

    # %% Data Format for Steel Dataset
    # df = pd.read_csv('UCI_Faults.csv')
    # df = pd.read_csv('Kaggle_Faults.csv')
    # features = df.iloc[:, :-7].values
    # labels = df.iloc[:, -7:].values
    # labels = np.argmax(labels, axis=1)

#%%
    # 标准化
    features = StandardScaler().fit_transform(features)

    # 生成掩码
    features, labels, train_mask, val_mask, test_mask = generate_masks(features, labels, train_ratio=0.8, seed=42)

    # 配置参数（与论文实验设置一致）
    config = {
        'model_type': 'GIN',
        'hidden_dim': 32,  # 隐藏层维度（论文默认16-64）
        'inner_lr': 0.01,  # 内层学习率（模型参数）
        'outer_lr': 0.1,  # 外层学习率（θ参数）
        'l2_reg': 5e-4,  # L2正则化系数
        'inner_steps': 10,  # 内层优化步数（论文τ参数）
        'max_epochs': 500,  # 最大训练轮次
        'patience': 20,  # 早停耐心值
        'graph_init': 'knn',  # 初始化方法
        'k': 10,  # kNN参数
        'num_layers': 2
    }

    # 训练LDS-GNN
    trainer = LDS_Trainer(features, labels, train_mask, val_mask, test_mask, config)
    trainer.train()