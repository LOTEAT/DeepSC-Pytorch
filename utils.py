'''
Author: LOTEAT
Date: 2023-05-31 16:10:57
'''  

import numpy as np
import torch


def snr2noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def create_padding_mask(seq):
    seq = torch.eq(seq, 0).float()
    return seq.unsqueeze(1).unsqueeze(2) 

def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask 

def create_masks(data, target):
    enc_padding_mask = create_padding_mask(data)
    dec_padding_mask = create_padding_mask(data)
    look_ahead_mask = create_look_ahead_mask(target.size(1))
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def sample_batch(data, noise):
    data = data.view(-1, 1)
    noise = noise.view(-1, 1)
    batch_data1, batch_data2 = torch.split(data, data.size(0) // 2, dim=0)
    batch_noise1, batch_noise2 = torch.split(noise, noise.size(0) // 2, dim=0)
    joint = torch.cat([batch_data1, batch_noise1], dim=1)
    marginal = torch.cat([batch_data1, batch_noise2], dim=1)
    return joint, marginal

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et
