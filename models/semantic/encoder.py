'''
Author: LOTEAT
Date: 2023-06-06 20:08:31
'''
import torch.nn as nn
from ..attention import MultiHeadedAttention
from .embedding import Embeddings
from .feedforward import PositionwiseFeedForward
from .sublayer import SublayerConnection
from .utils import position_encoding

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, d_model, num_heads, dff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model)
        self.sublayer1 = SublayerConnection(size, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dff)
        self.sublayer2 = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)  
        attn_output = self.sublayer1(x, attn_output)
        ffn_output = self.feed_forward(attn_output)
        ffn_output = self.sublayer2(attn_output, ffn_output)
        return ffn_output

class SemanticEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        dff,
        input_vocab_size,
        maximum_position_encoding=512,
        dropout_pro=0.1,
    ):
        super(SemanticEncoder, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_layers = num_layers
        self.target_vocab_size = input_vocab_size
        self.embedding = Embeddings(d_model, input_vocab_size)
        self.pos_encoding = position_encoding(maximum_position_encoding, self.d_model)
        self.dropout = nn.Dropout(dropout_pro)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, d_model, num_heads, dff, dropout_pro)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        self.pos_encoding = self.pos_encoding.to(x.device)
        seq_len = x.shape[1]
        # Embedding
        x = self.embedding(x)
        # positional Encoding
        x += self.pos_encoding[:, :seq_len, :]
        # Dropout
        x = self.dropout(x)
        # Encoder
        
        for i in range(self.num_layers):
            x = self.encoder[i](x, mask)
        return x