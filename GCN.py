import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.layer2 = GraphConv(hidden_dim, out_dim, activation=None)
        self.dropout = dropout

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(g, h)
        return h

    def extract_embeddings(self, g, h):
        h = self.layer1(g, h)
        return h.view(h.size(0), -1)
