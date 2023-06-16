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
    return seq.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tar.size(1))
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def sample_batch(rec, noise):
    rec = rec.view(-1, 1)
    noise = noise.view(-1, 1)
    rec_sample1, rec_sample2 = torch.split(rec, rec.size(0) // 2, dim=0)
    noise_sample1, noise_sample2 = torch.split(noise, noise.size(0) // 2, dim=0)
    joint = torch.cat([rec_sample1, noise_sample1], dim=1)
    marg = torch.cat([rec_sample1, noise_sample2], dim=1)
    return joint, marg

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et
