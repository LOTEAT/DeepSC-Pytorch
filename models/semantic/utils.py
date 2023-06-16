'''
Author: LOTEAT
Date: 2023-06-16 15:57:45
'''
import numpy as np
import torch

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