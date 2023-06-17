'''
Author: LOTEAT
Date: 2023-06-16 16:40:21
'''
from torch import nn
from ..layer_norm import LayerNorm

class ChannelDecoder(nn.Module):
    def __init__(self, d_model):
        super(ChannelDecoder, self).__init__()
        self.dense1 = nn.Linear(16, d_model)
        self.ac_fun1 = nn.ReLU()
        self.dense2 = nn.Linear(d_model, 512)
        self.ac_fun2 = nn.ReLU()
        self.dense3 = nn.Linear(512, d_model)
        self.layernorm1 = LayerNorm(128)

    def forward(self, data):
        x1 = self.dense1(data)
        x1 = self.ac_fun1(x1)
        x2 = self.dense2(x1)
        x2 = self.ac_fun2(x2)
        x3 = self.dense3(x2)
        output = self.layernorm1(x1 + x3)
        return output