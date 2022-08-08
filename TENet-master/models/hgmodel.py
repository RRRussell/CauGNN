from hglayer import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.MultiHeadAttention import MultiheadAttention, ScaledDotProductSelfAttention

import torch

from utils import slicing
import numpy as np
from dgl.nn.pytorch import GATConv
# from attention import MultiheadAttention
from utils import slicing

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            print("hello world")
            print(len(num_heads))
            print(num_heads)
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

class GCN_multirelation(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode="none", attention_mode="none"):
        super(GCN_multirelation, self).__init__()

        self.gc1 = GraphConvolution(num_relation, num_entities, num_adjs, nfeat, nhid, attention_mode=attention_mode)
        self.gc2 = GraphConvolution(num_relation, num_entities, num_adjs, self.gc1.out_features, nhid, attention_mode=attention_mode)
        self.dropout = dropout
        if skip_mode not in ["add", "concat", "none"]:
            print("skip mode {} unknown, use default option 'none'".format(skip_mode))
            skip_mode = "add"
        elif skip_mode in ["concat"]:
            self.ff = nn.Linear(self.gc1.out_features + self.gc2.out_features, self.gc2.out_features)
        self.skip_mode = skip_mode
        self.out_dim = self.gc2.out_features

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode=="concat" else x2+x1

    def forward(self, x, adjs):
        x1 = F.relu(self.gc1(x, adjs))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adjs))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        return x2 if self.skip_mode is "none" else self.skip_connect_out(x2, x1)

class Classification(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nhid, features, out_dim, dropout, num_heads, regularization=None, gcn=None, skip_mode="none", attention_mode="none", trainable_features=None):
        super(Classification, self).__init__()
        # self.gcn = GCN_multirelation(num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode=skip_mode, attention_mode=attention_mode) if gcn is None else gcn
        self.han = HAN(num_adjs, features.shape[1], nhid, out_dim, num_heads, dropout)
        # self.classifier = nn.Linear(self.gcn.out_dim, nclass)
        self.reg_param = regularization if regularization else 0
        self.trainable_features = trainable_features if trainable_features else None

    def forward(self, x, adjs, calc_gcn=True):
        # x = self.gcn(x, adjs) if calc_gcn else x
        # x = self.classifier(x)
        x = self.han(adjs, x)
        x = F.log_softmax(x, dim=1)
        return x

    def regularization_loss(self, embedding):
        if not self.reg_param:
            return 0
        return self.reg_param * torch.mean(embedding.pow(2))

    def get_loss(self, output, labels, idx_lst):
        reg_loss = self.regularization_loss(output) # regularize the embeddings
        return F.nll_loss(output[idx_lst], labels[idx_lst]) + reg_loss
