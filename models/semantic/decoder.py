'''
Author: LOTEAT
Date: 2023-06-06 20:08:36
'''
import torch.nn as nn
import math
from ..attention import MultiHeadedAttention
from .feedforward import PositionwiseFeedForward
from .sublayer import SublayerConnection
from .utils import position_encoding

class DecoderLayer(nn.Module):
    """
    This is decoder leayer, which includes three layers,
    1. multihead,
    2. masked multihead
    3. feed forward
    """

    def __init__(self, size, d_model, num_heads, dff, drop_pro=0.1):
        super(DecoderLayer, self).__init__()
        
        self.attention_layer1 = MultiHeadedAttention(num_heads, d_model)  # masked
        self.attention_layer2 = MultiHeadedAttention(num_heads, d_model)

        self.ffn = PositionwiseFeedForward(d_model, dff)
        self.sublayer1 = SublayerConnection(size, drop_pro)
        self.sublayer2 = SublayerConnection(size, drop_pro)
        self.sublayer3 = SublayerConnection(size, drop_pro)
        self.feed_forward = PositionwiseFeedForward(d_model, dff)
        self.size = size


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights1 = self.attention_layer1(x, x, x, look_ahead_mask)
        output1 = self.sublayer1(x, attn1)
        attn2, attn_weights2 = self.attention_layer2(
            output1, enc_output, enc_output, padding_mask
        )
        
        output2 = self.sublayer2(output1, attn2)
        ffn_output = self.ffn(output2)
        output3 = self.sublayer3(output2, ffn_output)
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
        num_heads,
        d_model,
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

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, d_model, num_heads, dff, dropout_pro)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_pro)
        # prediction layer
        self.final_layer = nn.Linear(128, target_vocab_size)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        self.pos_encoding = self.pos_encoding.to(x.device)
        seq_len = x.shape[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )

        attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
        attention_weights["decoder_layer{}_block2".format(i + 1)] = block2
        x = self.final_layer(x)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
