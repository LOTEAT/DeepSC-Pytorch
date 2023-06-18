'''
Author: LOTEAT
Date: 2023-06-17 22:10:24
'''
from utils import create_masks, sample_batch, mutual_information
from loss import SparseCategoricalCrossentropyLoss

criterion = SparseCategoricalCrossentropyLoss()

def train(data, target, transceiver, mine_net, optim_net, optim_mi, channel='AWGN', n_std=0.1, use_mine=False):
    tar_inp = target[:, :-1]  # exclude the last one
    tar_real = target[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(data, tar_inp)

    optim_net.zero_grad()
    optim_mi.zero_grad()

    # Forward pass
    predictions, channel_enc_output, received_channel_enc_output = transceiver(
        data, tar_inp, channel=channel, n_std=n_std,
        enc_padding_mask=enc_padding_mask,
        combined_mask=combined_mask, dec_padding_mask=dec_padding_mask
    )
    # Compute loss
    loss_error = criterion(tar_real, predictions)
    loss = loss_error
    
    if use_mine:
        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)
        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb
        loss += 0.05 * loss_mine

    # Compute gradients and update network parameters
    loss.backward()
    optim_net.step()

    if use_mine:
        # Compute gradients and update MI estimator parameters
        optim_mi.zero_grad()
        loss_mine.backward()
        optim_mi.step()

    mi_numerical = 2.20  # Placeholder value, update with actual value

    return loss, None, mi_numerical