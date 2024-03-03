import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge


class GCN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNConv(7, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 256)
        self.lin3 = Linear(256, 30)
        self.contest = torch.nn.Conv2d()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data[4], data[3], data[7]
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        feature = self.lin2(x)
        x = self.lin3(feature)
        return F.log_softmax(x, dim=-1), feature

    def __repr__(self):
        return self.__class__.__name__


