import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
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
        self.dropout = dropout

    def forward(self, g, h):
        h = self.embedding(h)
        for conv in self.layers:
            h = conv(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.readout(h)
        return h

    def extract_embeddings(self, g, h):
        h = self.embedding(h)
        return h
