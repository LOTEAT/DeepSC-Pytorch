'''
Author: LOTEAT
Date: 2023-06-06 19:28:31
'''
import math
import torch
import torch.nn as nn

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9) 
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        assert d_model % self.num_heads == 0
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.dense = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):    
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for lin, x in zip((self.wq, self.wk, self.wv), (query, key, value))
        ]
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_heads * self.head_dim)
        )
        return self.dense(x), self.attn