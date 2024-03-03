import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from .base_mdel import BaseModel
from torch_geometric.nn import (SAGEConv, global_mean_pool,
                                JumpingKnowledge)
from .edge_pool import EdgePooling


class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')
        self.lin_node = SAGEConv(in_channels, out_channels)
        self.lin_edge = torch.nn.Linear(in_channels, 7)

    def forward(self, x, edge_index, edge_attr):

        edge_attr = self.lin_edge(edge_attr)
        x = self.lin_node(x)
        x = x * edge_attr

        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return F.relu(aggr_out)

class SageEdge(BaseModel):
    def __init__(self, name, num_layers=1, hidden=256):
        super(SageEdge, self).__init__(name)
        self.edge_lin = Linear(4, hidden)
        self.conv1 = CustomConv(7, hidden)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            CustomConv(hidden, hidden)
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin_final = torch.nn.Linear(hidden * 2, 30)
        self.lin2 = Linear(hidden, 30)

    def reset_parameters(self):

        self.edge_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # x, edge_index, batch = data[4], data[3], data[7]
        x, edge_index, batch, edge_attr = data[4].to(device='cuda'), data[3].to(device='cuda'), data[7].to(device='cuda'), data[8].to(device='cuda')

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, batch, edge_attr = pool(x, edge_index, batch=batch, edge_attr=edge_attr)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        feature = x
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), feature

    def __repr__(self):
        return self.__class__.__name__
