import os
import os.path as osp
import glob
import sys

import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.io.txt_array import read_txt_str
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

names = [
    'A', 'graph_indicator', 'node_labels', 'edge_labels', 'node_attributes',
    'edge_attributes', 'graph_labels', 'graph_attributes', 'graph_names'
]


def read_tu_data(folder):
    files = glob.glob(osp.join(folder, '*.txt'))#得到数据集文件夹下的txt文件
    names = [f.split('\\')[-1].split('.')[0] for f in files]#分割文件，得到文件名

    edge_index = read_file(folder, 'A', torch.long).t() - 1 #读取边的对应，并转换格式,-1为了从0开始，与索引对应
    batch = read_file(folder, 'graph_indicator', torch.long) - 1  #读取节点对应的图编号

    # node_attributes = read_file(folder, 'node_attributes')
    # x = node_attributes
    node_attributes = node_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, 'node_attributes') #结点属性，7维向量 草图几何属性(面类型，面法向量坐标，面切向量坐标)
    if 'node_labels' in names:
        node_labels = read_file(folder, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attributes, node_labels])  # X为节点属性和标签拼接在一起的二维数组
    print('x:',x.shape)
    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, 'edge_attributes') #边的属性，4维向量，边属性，边类型，边方向，边长度
    if 'edge_labels' in names:
        edge_labels = read_file(folder, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])  # edge_attr为边的属性和标签拼接在一起的二维数组

    y = None
    name, name_dict = read_file(folder, 'graph_names', isName=True)  #得到图编号和对应的字典
    y = read_file(folder, 'graph_labels', torch.long)  #得到类别标签
    _, y = y.unique(sorted=True, return_inverse=True)  #将得到的y进行unique操作，将原数组元素在去重数组的索引数组返回给y
    print('y:',y.shape)
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)  #得到节点个数
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr) #删除自环
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)  #对相同索引的多个值求和
    print('edge_index:', edge_index.shape)
    print('edge_attr:', edge_attr.shape)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name=name, name_dict=name_dict)
    data, slices = split(data, batch)
    return x, edge_index, y, name, name_dict, slices, data.__num_nodes__, edge_attr


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    '''
        创建节点索引和边索引
        np.bincount 统计从0到数组最大元素出现个数
        torch.from_numpy 将数组转换成tensor
        torch.cumsum 将张量元素累加
    '''
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
            slices['name'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
            slices['name'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices

#folder 文件夹地址  name 具体的txt dtype 类型  isName
def read_file(folder, name, dtype=None, isName=False):
    if not isName:
        path = osp.join(folder, '{}.txt'.format(name))
        # read_txt_array 读取txt文件的数据，并将其转换成列表
        return read_txt_array(path, sep=',', dtype=dtype)
    else:
        path = osp.join(folder, '{}.txt'.format(name))
        return read_txt_str(path, sep=',')


def write_to_txt(path, data):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            print(f)

    with open(path, 'a', encoding='utf-8') as f:
        f.write(data)


def normalization(data):
    _range = data.max()
    return data / _range