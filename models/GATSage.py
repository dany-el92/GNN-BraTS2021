import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv, ChebConv
from dgl.nn.pytorch.conv import SAGEConv

'''
Contains the actual neural network architectures.
Supports GraphSAGE with either the pool,mean,gcn, or lstm aggregator as well as GAT.
The input, output, and intermediate layer sizes can all be specified.
Typically will call init_graph_net and pass along the desired model and hyperparameters.

Also contains the CNN Refinement net which is a very simple 2 layer 3D convolutional neural network.
As input, it expects 8 channels, which are the concatenated 4 input modalities and 4 output logits of the GNN predictions.
'''


class GraphSage(nn.Module):
    def __init__(self,in_feats,layer_sizes,n_classes,aggregator_type,dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, layer_sizes[0], aggregator_type, feat_drop=dropout, activation=F.relu))
        # hidden layers
        for i in range(1,len(layer_sizes)):
            self.layers.append(SAGEConv(layer_sizes[i-1], layer_sizes[i], aggregator_type, feat_drop=dropout, activation=F.relu))
        # output layer
        self.layers.append(SAGEConv(layer_sizes[-1], n_classes, aggregator_type, feat_drop=0, activation=None))

    def forward(self,graph,feat, **kwargs):
        h = feat
        for layer in self.layers:
            h = layer(graph, h)
        return h


class GAT(nn.Module):
    def __init__(self,in_feats,layer_sizes,n_classes,heads,residuals,
                activation=F.elu,feat_drop=0,attn_drop=0,negative_slope=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.layers.append(GATConv(
            in_feats, layer_sizes[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(1, len(layer_sizes)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                layer_sizes[i-1] * heads[i-1], layer_sizes[i], heads[i],
                feat_drop, attn_drop, negative_slope, residuals[i], self.activation, allow_zero_in_degree=True))
        # output projection
        self.layers.append(GATConv(
            layer_sizes[-1] * heads[-1], n_classes, 1,
            feat_drop, attn_drop, negative_slope, False, None, allow_zero_in_degree=True))

    def forward(self,g, inputs):
        h = inputs
        for l in range(len(self.layers)-1):
            h = self.layers[l](g, h).flatten(1)
        # output projection
        logits = self.layers[-1](g, h).mean(1)
        return logits


from dgl.nn.pytorch import GINConv
import torch.nn.functional as F

class GIN(nn.Module):
    def __init__(self, in_feats, layer_sizes, n_classes, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # input layer
        self.layers.append(GINConv(apply_func=nn.Linear(in_feats, layer_sizes[0]), aggregator_type='sum'))
        # hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(GINConv(apply_func=nn.Linear(layer_sizes[i-1], layer_sizes[i]), aggregator_type='sum'))
        # output layer
        self.layers.append(GINConv(apply_func=nn.Linear(layer_sizes[-1], n_classes), aggregator_type='sum'))

    def forward(self, g, feat):
        h = feat
        for layer in self.layers[:-1]:
            h = layer(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](g, h)
        return h


class ChebNet(nn.Module):
    def __init__(self, in_feats, layer_sizes, n_classes, k, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        # Input layer
        self.layers.append(ChebConv(in_feats, layer_sizes[0], k))

        # Hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(ChebConv(layer_sizes[i-1], layer_sizes[i], k))

        # Output layer
        self.layers.append(ChebConv(layer_sizes[-1], n_classes, k))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1: # No activation and dropout on the output layer
                h = F.relu(h)
                h = self.dropout(h)
        return h