'''
Author: LOTEAT
Date: 2023-05-30 21:23:59
'''
import torch
from torch import nn
import math
from .utils import LayerNorm

class PowerNorm(torch.nn.Module):
    def forward(self, x):
        return x / torch.sqrt(2 * torch.mean(x ** 2))


class ChannelEncoder(nn.Module):
    def __init__(self, size1=256, size2=16):
        super(ChannelEncoder, self).__init__()

        self.dense0 = nn.Linear(size1, size1)
        self.ac_fun1 = nn.ReLU()
        self.dense1 = nn.Linear(size1, size2)
        self.powernorm = PowerNorm()

    def forward(self, inputs):
        outputs1 = self.dense0(inputs)
        outputs1 = self.ac_fun1(inputs)
        outputs2 = self.dense1(outputs1)
        # POWER = tf.sqrt(tf.reduce_mean(tf.square(outputs2)))
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs


class ChannelDecoder(nn.Module):
    def __init__(self, size1, size2):
        super(ChannelDecoder, self).__init__()
        self.dense1 = nn.Linear(size1, size1)
        self.ac_fun1 = nn.ReLU()
        self.dense2 = nn.Linear(size1, size2)
        self.ac_fun2 = nn.ReLU()
        # size2 equals to d_model
        self.dense3 = nn.Linear(size2, size1)

        self.layernorm1 = LayerNorm(5)

    def forward(self, receives):
        x1 = self.dense1(receives)
        x1 = self.ac_fun1(x1)
        x2 = self.dense2(x1)
        x2 = self.ac_fun2(x2)
        x3 = self.dense3(x2)

        output = self.layernorm1(x1 + x3)
        return output
    
    
class Channel(nn.Module):
    def __init__(self):
        super(Channel, self).__init__()
    
    def awgn(inputs, n_std=0.1):
        x = inputs
        y = x + torch.randn_like(x) * n_std
        return y

    def fading(inputs, K=1, n_std=0.1, detector='MMSE'):
        x = inputs
        bs, sent_len, d_model = x.shape
        mean = math.sqrt(K / (2 * (K + 1)))
        std = math.sqrt(1 / (2 * (K + 1)))
        x = x.view(bs, -1, 2)
        x_real = x[:, :, 0]
        x_imag = x[:, :, 1]
        x_complex = torch.complex(x_real, x_imag)

        # create the fading factor
        h_real = torch.randn(1) * std + mean
        h_imag = torch.randn(1) * std + mean
        h_complex = torch.complex(h_real, h_imag)
        # create the noise vector
        n = torch.randn_like(x) * n_std
        n_real = n[:, :, 0]
        n_imag = n[:, :, 1]
        n_complex = torch.complex(n_real, n_imag)
        # Transmit Signals
        y_complex = x_complex * h_complex + n_complex
        # Employ the perfect CSI here
        if detector == 'LS':
            h_complex_conj = torch.conj(h_complex)
            x_est_complex = y_complex * h_complex_conj / (h_complex * h_complex_conj)
        elif detector == 'MMSE':
            # MMSE Detector
            h_complex_conj = torch.conj(h_complex)
            a = h_complex * h_complex_conj + (n_std * n_std * 2)
            x_est_complex = y_complex * h_complex_conj / a
        else:
            raise ValueError("detector must be 'LS' or 'MMSE'")
        x_est_real = torch.real(x_est_complex)
        x_est_img = torch.imag(x_est_complex)

        x_est_real = x_est_real.unsqueeze(-1)
        x_est_img = x_est_img.unsqueeze(-1)

        x_est = torch.cat([x_est_real, x_est_img], dim=-1)
        x_est = x_est.view(bs, sent_len, -1)

        # method 1
        noise_level = n_std * torch.ones(bs, sent_len, 1)
        h_real = h_real * torch.ones(bs, sent_len, 1)
        h_imag = h_imag * torch.ones(bs, sent_len, 1)
        h = torch.cat((h_real, h_imag), dim=-1)
        out1 = torch.cat((h, x_est), dim=-1)   # [bs, sent_len, 2 + d_model]

        # method 2
        y_complex_real = torch.real(y_complex)
        y_complex_img = torch.imag(y_complex)
        y_complex_real = y_complex_real.unsqueeze(-1)
        y_complex_img = y_complex_img.unsqueeze(-1)
        y = torch.cat([y_complex_real, y_complex_img], dim=-1)
        y = y.view(bs, sent_len, -1)
        out2 = torch.cat((h, y), dim=-1)  # [bs, sent_len, 2 + d_model]

        return x_est


