import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.sage_conv_edge_pool import SageEdge
from model.multi_view_conv import MVCNN
from model.base_mdel import BaseModel
from torch.autograd import Variable
from torch_sparse import SparseTensor, cat
from torch import Tensor
from torch.nn import Linear
from model.attention import MultiHead


class WrapperModel(BaseModel):
    def __init__(self, name, model1=MVCNN, model2=SageEdge):
        super(WrapperModel, self).__init__(name)
        self.model1 = MVCNN('m')
        self.model2 = SageEdge('s')

        self.mha1 = MultiHead(256, 256, heads=8)
        self.lin = Linear(512, 256)
        self.classific = Linear(256, 30)

    def forward(self, data):
        m_out = self.model1(data[1])[1]
        s_out = self.model2(data[4], data[3], data[7])[1]
        unsqueeze_s_out = s_out.unsqueeze(1)

        ps_m2s = self.mha1(unsqueeze_s_out, m_out, m_out).squeeze()
        enhance_s_out = torch.add(s_out, torch.mul(s_out, ps_m2s))

        feature = self.lin(torch.cat([enhance_s_out, torch.max(m_out, 1)[0].view(8, -1)], 1))
        out = F.log_softmax(self.classific(feature), -1)
        cor = np.linalg.norm(torch.sub(F.sigmoid(torch.log(torch.abs(enhance_s_out))), F.sigmoid(torch.log(torch.abs(torch.max(m_out, 1)[0])))))
        return out, feature, cor


def my_collate_fun(data):
    new_data = [[], [], [], [], [], [], [], []]
    for old in data:
        for i in range(8):
            new_data[i].append(old[i])
    data = new_data

    x = torch.stack(tuple([img for img in data[1]]), 0)
    N, V, C, H, W = x.size()
    x = Variable(x).view(-1, C, H, W)
    data[1] = x

    batch = []
    for i, items in enumerate(data[7]):
        num_nodes = items[0]
        batch.append(torch.full((num_nodes,), i, dtype=torch.long))
    data[7] = batch

    for i, items in enumerate(data):
        if i == 1 or i == 2:
            continue

        item = items[0]
        if i == 3:
            cat_dim = -1
        else:
            cat_dim = 0

        if isinstance(item, Tensor):
            data[i] = torch.cat(items, cat_dim)
        elif isinstance(item, SparseTensor):
            data[i] = cat(items, cat_dim)
        elif isinstance(item, (int, float)):
            data[i] = torch.tensor(items)

    return data
