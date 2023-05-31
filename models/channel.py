'''
Author: LOTEAT
Date: 2023-05-30 21:23:59
'''
import torch
from torch import nn

class PowerNorm(torch.nn.Module):
    def forward(self, x):
        return x / torch.sqrt(2 * torch.mean(x ** 2))

self.powernorm = PowerNorm()



class ChannelEncoder(nn.Module):
    def __init__(self, size1=256, size2=16):
        super(ChannelEncoder, self).__init__()

        self.dense0 = nn.Linear(size1, activation="relu")
        self.dense1 = nn.Linear(size2, activation=None)
        self.powernorm = PowerNorm()

    def forward(self, inputs):
        outputs1 = self.dense0(inputs)
        outputs2 = self.dense1(outputs1)
        # POWER = tf.sqrt(tf.reduce_mean(tf.square(outputs2)))
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs


class ChannelDecoder(nn.Module):
    def __init__(self, size1, size2):
        super(ChannelDecoder, self).__init__()
        self.dense1 = nn.Linear(size1, activation="relu")
        self.dense2 = nn.Linear(size2, activation="relu")
        # size2 equals to d_model
        self.dense3 = nn.Linear(size1, activation=None)

        self.layernorm1 = nn.LayerNorm(eps=1e-6)

    def forward(self, receives):
        x1 = self.dense1(receives)
        x2 = self.dense2(x1)
        x3 = self.dense3(x2)

        output = self.layernorm1(x1 + x3)
        return output
