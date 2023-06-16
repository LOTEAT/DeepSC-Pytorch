'''
Author: LOTEAT
Date: 2023-06-16 16:38:17
'''
from torch import nn
from powernorm import PowerNorm


class ChannelEncoder(nn.Module):
    def __init__(self, size1=256, size2=16):
        super(ChannelEncoder, self).__init__()
        self.dense0 = nn.Linear(128, 256)
        self.ac_fun1 = nn.ReLU()
        self.dense1 = nn.Linear(size1, size2)
        self.powernorm = PowerNorm()

    def forward(self, inputs):

        outputs1 = self.dense0(inputs)
        outputs1 = self.ac_fun1(outputs1)
        outputs2 = self.dense1(outputs1)
        # POWER = tf.sqrt(tf.reduce_mean(tf.square(outputs2)))
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs