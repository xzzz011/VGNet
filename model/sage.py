import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class EdgeWeightedSAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels, edge_attr_dim, normalize=True, bias=True):
        super(EdgeWeightedSAGEConv, self).__init__(in_channels, out_channels, normalize=normalize, bias=bias)
        self.edge_lin = nn.Linear(edge_attr_dim, in_channels)
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        edge_weights = torch.sigmoid(self.edge_lin(edge_attr))

        row, col = edge_index
        weighted_x = x[col] * edge_weights.view(-1, 1)
        aggr_x = self.aggr(weighted_x, row, col, x.size(self.node_dim))

        self_weight = torch.ones(x.size(self.node_dim), 1).to(x.device)
        weighted_x_self = x * self_weight
        aggr_x += F.linear(weighted_x_self, self.weight, self.bias)

        if self.normalize:
            aggr_x = F.normalize(aggr_x, p=2, dim=-1)

        return aggr_x
