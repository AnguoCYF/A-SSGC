import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from GCN import *
from GIN import *
from GAT import *

from A_SSGC import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% 读取数据构建图
df = pd.read_csv('glass.csv')
features = df.drop(columns=['Type']).values
unique_labels = df['Type'].unique()
label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
df['Type'] = df['Type'].map(label_mapping)
labels = df['Type'].values
num_classes = len(np.unique(labels))
scaler = StandardScaler()
features = scaler.fit_transform(features)

class_counts = np.bincount(labels)
class_weights = len(labels) / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

#%% 初始化GNN分类器
def train(g, model, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

    best_f1_test=0
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

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.3f},Train Accuracy: {acc_train:.3f}, Test Accuracy: {acc_test:.3f},"
                  f"Test Precision: {prec_test:.3f},Test Recall: {recall_test:.3f},Test F1: {f1_test:.3f}")

    best_acc_test, best_prec, best_recall, best_f1 = best_metrics
    print(f"\nBest Test Accuracy at Epoch {best_epoch+1}: Accuracy: {best_acc_test:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

def evaluate(g, model, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        logits = logits[mask]
        labels = g.ndata['label'][mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        precision = precision_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    return acc, precision, recall, f1

#%%
in_dim = features.shape[1]
train_ratio = 0.8
random_state = 42

hidden_dim = 320
epochs = 500
lr = 0.005
dropout = 0.1
base_thres = 0.35
degree_factor = 0.25

_, _, train_mask, test_mask = generate_masks(features, labels, train_ratio, seed=random_state,
                                                         resample=False)

# Graph Construction: changed the hyperparm 'method' to select different methods
g = build_dgl_graph(features, labels, method='fuse', param=[base_thres, degree_factor]).to(device)

# 将掩码添加到图的ndata中
g.train_mask = train_mask.to(device)
g.test_mask = test_mask.to(device)

# GCN Model
model = GCN(in_dim, hidden_dim, out_dim=num_classes, dropout=dropout).to(device)

# num_heads = 15 # GAT
# model = GAT(in_dim, hidden_dim,  num_heads=num_heads, out_dim=num_classes, dropout=dropout).to(device)

# num_layers = 3 # GIN
# model = GIN(in_dim, hidden_dim, out_dim=num_classes, num_layers=num_layers, dropout=dropout).to(device)

train(g, model, epochs, lr)
