'''
Author: LOTEAT
Date: 2023-05-30 21:28:34
'''
import torch
from torch import nn
import numpy as np

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
    return torch.from_numpy(pos_encoding, dtype = torch.float32)

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
    
        assert d_model % self.num_heads == 0

        # TODO
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def scale_dot_product_attention(self, q, k, v, mask):
        """
        softmax(Q*K^t/\sqrt(d_k))*V
        where the dimension of Q is [, seq_len_q, depth]
        the dimension of K is [, seq_len_k, depth]
        the dimension of V is  [, seq_len_v, depth_v]
        Q*K^t = [, seq_len_q, seq_len_k]
    
        Notice seq_len_k = seq_len_v
    
        mask uses in the Q*K^T ...[, seq_len_q, seq_len_k]
    
        the output is [, seq_len_q, depth_v]
        """
        attn_weights = torch.matmul(q, k.t()) / np.sqrt(self.d_model)
        if mask is not None:
            attn_weights += (mask * -1e9) 
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        outputs = torch.matmul(attn_weights, v)
        return outputs, attn_weights
    
    def split_heads(self, x):
        bs = x.shape[0]
        length = x.shape[1]
        depth = (self.d_model // self.num_heads)
        x = x.view([bs, length, self.num_heads, depth])
        return x.permute(0, 2, 1, 3)
    
    def combined_heads(self, x):
        bs = x.shape[0]
        length = x.shape[1]
        # [batch, length, num_heads, depth]
        x = x.permutate([0, 2, 1, 3]) 
        return x.reshape([bs, length, self.d_model]) 
    
    def forward(self, v, k, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q) # (batch_size, num_heads, seq_len_q, depth) where d_model = num_heads*depth
        k = self.split_heads(k) # (batch_size, num_heads, seq_len_k, depth) where d_model = num_heads*depth
        v = self.split_heads(v) # (batch_size, num_heads, seq_len_v, depth) where d_model = num_heads*depth
        
        scaled_attention, attention_weights = self.scale_dot_product_attention(q, k, v, mask)
        #[batch_size, seq_len_q, depth_v]
        attention_output = self.combined_heads(scaled_attention)
                
        multi_outputs = self.dense(attention_output)
        return multi_outputs, attention_weights
        

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dff = d_ff
    
    def point_wise_feed_forward_network(d_model, dff):
        '''
        This is point_wise_feed_forward_network
        FFN = max(0, x*W_1 + b_1)*W_2 +b_2
        '''
        return nn.ModuleList([
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        ])

        
        
            

class EncoderLayer(nn.Module):
    '''
    This is encoder layer, which includes two sublayers, multihead and feed forward.
    '''
    def __init__(self, d_model, num_heads, dff, drop_pro = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.attention_layer = AttentionLayer(d_model, num_heads)
        self.feed_layer = FeedForward(d_model, dff)
        
        self.layernorm1 = nn.LayerNorm(eps=1e-6)
        self.layernorm2 = nn.LayerNorm(epsilon=1e-6)
        
        self.dropout1 = nn.Dropout(drop_pro)
        self.dropout2 = nn.Dropout(drop_pro)
        
    def forward(self, x, training, mask):
        # attention: the layernorm(x + sublayer(x)) should be replaced by 
        # x + sublayer(LayerNorm(x)) 
        attn_output, _ = self.attention_layer(x, x, x, mask) #这个地方有点问题
        attn_output = self.dropout1(attn_output, training = training)
        output1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.feed_layer(output1)
        ffn_output = self.dropout2(ffn_output, training = training)
        output2 = self.layernorm2(output1 + ffn_output) # (batch_size, input_seq_len, d_model)
        
        return output2





class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dff, input_vocab_size, 
             maximum_position_encoding=512, dropout_pro=0.1):
        self.d_model = d_model
        self.dff = dff
        self.num_layers = num_layers
        self.target_vocab_size = input_vocab_size
        self.embedding = nn.Embedding(input_vocab_size, d_model) 
        self.pos_encoding = position_encoding(maximum_position_encoding, self.d_model)
        self.encoder = [EncoderLayer(d_model, num_heads, dff, dropout_pro) for _ in range(num_layers)]

        self.dropout = nn.Dropout(dropout_pro)
    
    def forward(self, x, training, mask):
        seq_len = x.shape[1]
        
        # Embedding
        x = self.embedding(x) 
        x *= torch.sqrt(self.d_model)
        # positional Encoding
        x += self.pos_encoding[:, :seq_len, :]
        
        # Dropout
        x = self.dropout(x, training = training)
        
        # Encoder
        for i in range(self.num_layers):
            x = self.encoder[i](x, training, mask)
            
        return x