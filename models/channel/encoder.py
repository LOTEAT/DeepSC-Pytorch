'''
Author: LOTEAT
Date: 2023-06-16 16:38:17
'''
from torch import nn
from .powernorm import PowerNorm


class ChannelEncoder(nn.Module):
    def __init__(self, d_model=128):
        super(ChannelEncoder, self).__init__()
        self.dense0 = nn.Linear(d_model, 256)
        self.ac_fun1 = nn.ReLU()
        self.dense1 = nn.Linear(256, 16)
        self.powernorm = PowerNorm()

    def forward(self, data):
        out = self.dense0(data)
        out = self.ac_fun1(out)
        out = self.dense1(out)
        out = self.powernorm(out)
        return out