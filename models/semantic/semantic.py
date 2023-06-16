"""
Author: LOTEAT
Date: 2023-05-30 21:28:34
"""
import torch
from torch import nn
import numpy as np
import math
from ..utils import LayerNorm
from ..attention import MultiHeadedAttention


def position_encoding(position, d_model):
    """
    Position encoder layer
    2i-th: sin(pos/10000^(2i/d_model))
    2i+1-th: cos(pos/10000^(2i/d_model))
    """
    pos = np.arange(position)[:, None]
    index = np.arange(d_model)[None, :]
    angle_set = pos / np.power(10000, ((2 * index) / np.float32(d_model)))
    # 2i
    angle_set[:, 0::2] = np.sin(angle_set[:, 0::2])
    # 2i+1
    angle_set[:, 1::2] = np.cos(angle_set[:, 1::2])
    pos_encoding = angle_set[None, ...]
    return torch.from_numpy(pos_encoding)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.ac_fun = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.ac_fun(self.w_1(x)))


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
    

