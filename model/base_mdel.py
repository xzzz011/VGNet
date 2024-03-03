import numpy as np
import os
import torch
import torch.nn as nn
import joblib


class BaseModel(nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()
        self.name = name

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self, path + '/{}.pkl'.format(name))


