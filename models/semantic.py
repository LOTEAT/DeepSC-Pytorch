"""
Author: LOTEAT
Date: 2023-05-30 21:28:34
"""
import torch
from torch import nn
import numpy as np
import math
from .utils import LayerNorm


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



def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


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
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        q = self.wq(q)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip((self.wq, self.wk, self.wv), (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        return self.dense(x), self.attn

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
        "Follow Figure 1 (left) for connections."
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
        self.embedding = Embeddings(input_vocab_size, d_model)
        self.pos_encoding = position_encoding(maximum_position_encoding, self.d_model)
        self.dropout = nn.Dropout(dropout_pro)
        self.encoder = [
            EncoderLayer(d_model, d_model, num_heads, dff, dropout_pro)
            for _ in range(num_layers)
        ]

    def forward(self, x, training, mask):
        seq_len = x.shape[1]
        # Embedding
        x = self.embedding(x)
        # positional Encoding
        x += self.pos_encoding[:, :seq_len, :]
        # Dropout
        x = self.dropout(x, training=training)
        # Encoder
        for i in range(self.num_layers):
            x = self.encoder[i](x, training, mask)

        return x


class DecoderLayer(nn.Module):
    """
    This is decoder leayer, which includes three layers,
    1. multihead,
    2. masked multihead
    3. feed forward
    """

    def __init__(self, d_model, num_heads, dff, drop_pro=0.1):
        super(DecoderLayer, self).__init__()

        self.attention_layer1 = AttentionLayer(d_model, num_heads)  # masked
        self.attention_layer2 = AttentionLayer(d_model, num_heads)

        self.ffn = FeedForward(d_model, dff)

        self.layernorm1 = nn.LayerNorm(eps=1e-6)
        self.layernorm2 = nn.LayerNorm(eps=1e-6)
        self.layernorm3 = nn.LayerNorm(eps=1e-6)

        self.dropout1 = nn.Dropout(drop_pro)
        self.dropout2 = nn.Dropout(drop_pro)
        self.dropout3 = nn.Dropout(drop_pro)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights1 = self.attention_layer1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        output1 = self.layernorm1(x + attn1)

        attn2, attn_weights2 = self.attention_layer2(
            enc_output, enc_output, output1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        output2 = self.layernorm2(attn2 + output1)

        ffn_output = self.ffn(output2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output3 = self.layernorm3(
            ffn_output + output2
        )  # (batch_size, target_seq_len, d_model)

        return output3, attn_weights1, attn_weights2


class SemanticDecoder(nn.Module):
    """
    1. Output Embedding
    2. Positional Encoding
    3. N decoder layers
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding=512,
        dropout_pro=0.1,
    ):
        super(SemanticDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = position_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_pro)
            for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(dropout_pro)
        # prediction layer
        self.final_layer = nn.Linear(target_vocab_size, target_vocab_size)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= torch.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

        attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
        attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        x = self.final_layer(x)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
