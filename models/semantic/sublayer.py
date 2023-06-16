'''
Author: LOTEAT
Date: 2023-06-16 15:47:35
'''

from torch import nn 
from ..layer_norm import LayerNorm

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_out):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(attn_out))
    