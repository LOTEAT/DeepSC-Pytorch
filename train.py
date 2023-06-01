'''
Author: LOTEAT
Date: 2023-05-31 15:56:52
'''
import os
import torch
from helper import helper
import json
from utils import *
from dataset import EuroparlDataset
from torch.utils.data import DataLoader
from models.transceiver import Transceiver, Mine
import torch.optim as optim
from loss import SparseCategoricalCrossentropyLoss
import torch
import torch.nn.functional as F

# torch.set_num_threads(n)
# import json
# import tensorflow as tf
# from models.transceiver import Transeiver, Mine
# from dataset.dataloader import return_loader
# from utlis.trainer import train_step, eval_step

def train_step(inp, tar, net, mine_net, optim_net, optim_mi, channel='AWGN', n_std=0.1, train_with_mine=False):
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    optim_net.zero_grad()
    optim_mi.zero_grad()

    # Forward pass
    predictions, channel_enc_output, received_channel_enc_output = net(
        inp, tar_inp, channel=channel, n_std=n_std,
        enc_padding_mask=enc_padding_mask,
        combined_mask=combined_mask, dec_padding_mask=dec_padding_mask
    )

    # Compute loss
    loss_error = SparseCategoricalCrossentropyLoss(tar_real, predictions)
    loss = loss_error

    if train_with_mine:
        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)
        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb
        loss += 0.05 * loss_mine

    # Compute gradients and update network parameters
    loss.backward()
    optim_net.step()

    if train_with_mine:
        # Compute gradients and update MI estimator parameters
        optim_mi.zero_grad()
        loss_mine.backward()
        optim_mi.step()

    mi_numerical = 2.20  # Placeholder value, update with actual value

    return loss, loss_mine, mi_numerical


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(5)
    args = helper()
    torch.set_num_threads(args.nthreads)
    # Load the vocab
    vocab = json.load(open(args.vocab_path, 'rb'))
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    StoT = Seq2Text(token_to_idx, args.end_idx)
    # Load dataset
    train_dataset, test_dataset = EuroparlDataset(args.train_save_path), EuroparlDataset(args.test_save_path)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
    
    transeiver = Transceiver(args)
    mine_net = Mine()     

    # Define the optimizer
    optim_net = optim.Adam(transeiver.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-8)
    optim_mi = optim.Adam(mine_net.parameters(), lr=0.001)

    
    # Training the model
    best_loss = 10
    for epoch in range(args.epochs):
        n_std = snr2noise(args.train_snr)
        train_loss_record, test_loss_record = 0, 0
        for (batch, (inp, tar)) in enumerate(train_dataset):
            print(inp.shape, tar.shape)
            input()
            train_loss, train_loss_mine, _ = train_step(inp, tar, net, mine_net, optim_net, optim_mi, args.channel, n_std,
                                            train_with_mine=args.train_with_mine)
            train_loss_record += train_loss
        train_loss_record = train_loss_record/batch

        # # Valid
        # for (batch, (inp, tar)) in enumerate(test_dataset):
        #     test_loss = eval_step(inp, tar, net, args.channel, n_std)
        #     test_loss_record += test_loss
        # test_loss_record = test_loss_record / batch

        # if best_loss > test_loss_record:
        #     best_loss = test_loss_record
        #     manager.save(checkpoint_number=epoch)

        # print('Epoch {} Train Loss {:.4f} Test Loss {:.4f}'.format(epoch + 1, train_loss_record, test_loss_record))

