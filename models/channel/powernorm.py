'''
Author: LOTEAT
Date: 2023-06-16 16:38:47
'''
import torch
from torch import nn

class PowerNorm(nn.Module):
    def __init__(self):
        super(PowerNorm, self).__init__()
        
    def forward(self, x):
        return x / torch.sqrt(2 * torch.mean(x ** 2))