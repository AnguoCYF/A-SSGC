import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import dgl

from GraphVAE import GraphVAE
from A_SSGC import generate_masks
from GIN import GIN
from GAT import GAT
from GCN import GCN


##########################################
# 2. 定义下游分类任务的评估和训练函数
##########################################
def evaluate(g, model, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        logits = logits[mask]
        labels = g.ndata['label'][mask]
        _, pred = torch.max(logits, dim=1)
        correct = torch.sum(pred == labels)
        acc = correct.item() / len(labels)
        precision = precision_score(labels.cpu(), pred.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), pred.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), pred.cpu(), average='macro', zero_division=0)
    return acc, precision, recall, f1


def train(g, model, epochs, lr, class_weights, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

    best_f1_test = 0
    best_epoch = 0
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        logits = model(g, g.ndata['feat'])
        loss = loss_fn(logits[g.train_mask], g.ndata['label'][g.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train, prec_train, recall_train, f1_train = evaluate(g, model, g.train_mask)
        acc_test, prec_test, recall_test, f1_test = evaluate(g, model, g.test_mask)

        if f1_test > best_f1_test:
            best_f1_test = f1_test
            best_epoch = epoch
            best_metrics = (acc_test, prec_test, recall_test, f1_test)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.3f}, "
                  f"Train Acc: {acc_train:.3f}, Test Acc: {acc_test:.3f}, "
                  f"Test Prec: {prec_test:.3f}, Test Recall: {recall_test:.3f}, Test F1: {f1_test:.3f}")

    best_acc_test, best_prec, best_recall, best_f1 = best_metrics
    print(f"\nBest Test Performance at Epoch {best_epoch + 1}: "
          f"Acc: {best_acc_test:.3f}, Prec: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")


##########################################
# 3. 主程序
##########################################
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # %% Data Format for glass Dataset
    # df = pd.read_csv('glass.csv')
    # features = df.drop(columns=['Type']).values
    # # 将标签转换为整数（使用 pd.factorize 保证标签连续）
    # labels, uniques = pd.factorize(df['Type'])
    # num_classes = len(uniques)

    # %% Data Format for Steel Dataset
    df = pd.read_csv('UCI_Faults.csv')
    # df = pd.read_csv('Kaggle_Faults.csv')

    features = df.iloc[:, :-7].values
    labels = df.iloc[:, -7:].values
    labels = np.argmax(labels, axis=1)
    num_classes = len(np.unique(labels))

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 使用 generate_masks 划分数据集
    features, labels, train_mask, test_mask = generate_masks(features, labels, train_ratio=0.8, seed=42)

    ##########################################
    # 利用 GraphVAE 进行图构建
    ##########################################
    num_nodes = features.shape[0]

    # 构建全连接图的初始邻接矩阵（节点数等于样本数）
    G_nx = nx.complete_graph(num_nodes)

    G_nx.add_edges_from([(i, i) for i in range(num_nodes)])  # 添加自环，即每个节点与自己有一条边
    init_adj = torch.tensor(nx.to_numpy_array(G_nx), dtype=torch.float32)

    # GraphVAE 输入准备：整个数据集视为一个图，batch_size = 1
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)  # shape: (1, num_nodes, input_dim)
    init_adj_tensor = init_adj.to(device).unsqueeze(0)  # shape: (1, num_nodes, num_nodes)

    # GraphVAE 超参数（可根据需要调整）
    input_dim = features.shape[1]
    latent_dim = 10
    vae = GraphVAE(input_dim, latent_dim, num_nodes).to(device)
    vae.train()

    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=0.005)
    epochs_vae = 100
    for epoch in range(epochs_vae):
        optimizer_vae.zero_grad()
        loss = vae(features_tensor, init_adj_tensor)
        loss.backward()
        optimizer_vae.step()
        if (epoch + 1) % 10 == 0:
            print(f"GraphVAE Epoch {epoch + 1}/{epochs_vae}, Loss: {loss.item():.4f}")

    vae.eval()
    # 生成优化后的邻接矩阵（二值化处理）
    optimized_adj = vae.generate(features_tensor)  # shape: (num_nodes, num_nodes)
    optimized_adj_np = optimized_adj.cpu().numpy()

    # 利用生成的邻接矩阵构建 DGL 图
    G_nx_optimized = nx.from_numpy_array(optimized_adj_np)
    g = dgl.from_networkx(G_nx_optimized).to(device)

    # 将节点特征、标签以及训练/测试掩码添加到图中
    g.ndata['feat'] = torch.tensor(features, dtype=torch.float32).to(device)
    g.ndata['label'] = torch.tensor(labels, dtype=torch.long).to(device)
    g.train_mask = train_mask.to(device)
    g.test_mask = test_mask.to(device)

    # dgl.save_graphs('GraphVAE_graph.dgl', [g])
    print(f'Graph Construction Complete.')
    ##########################################
    # 下游分类任务：优化classifier
    ##########################################
    # 计算类别权重（防止类别不均衡）
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    in_dim = features.shape[1]
    cl_hdim = 64
    dropout = 0.2
    epochs_gnn = 500
    lr_gnn = 0.005

    classifier = GCN(in_dim, cl_hdim, out_dim=num_classes, dropout=dropout).to(device)

    # num_heads = 8
    # classifier = GAT(in_dim, cl_hdim,  num_heads=num_heads, out_dim=num_classes, dropout=dropout).to(device)

    # num_layers = 1  # GIN
    # classifier = GIN(in_dim, cl_hdim, out_dim=num_classes, num_layers=num_layers, dropout=dropout).to(device)

    print("\nTraining GNN classfier on the constructed graph...\n")
    train(g, classifier, epochs_gnn, lr_gnn, class_weights, device)
