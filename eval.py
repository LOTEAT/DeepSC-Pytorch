import json
import numpy as np
import torch
from models.transceiver import Transceiver
from tools import SeqtoText, BleuScore, SNR_to_noise, Similarity
from utils import *
from helper import helper
from dataset import EuroparlDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def greedy_decode(args, inp, net, channel='AWGN', n_std=0.1):
    bs, sent_len = inp.shape
    # notice all of the test sentence add the <start> and <end>
    # using <start> as the start of decoder
    outputs = args.start_idx * torch.ones([bs, 1], dtype=torch.long).to('cuda')
    # strat decoding
    enc_padding_mask = create_padding_mask(inp)
    sema_enc_output = net.semantic_encoder(inp, enc_padding_mask)
    # channel encoder
    channel_enc_output = net.channel_encoder(sema_enc_output)
    # over the AWGN channel
    if channel == 'AWGN':
        received_channel_enc_output = net.channel.awgn(channel_enc_output, n_std)
    elif channel == 'Rician':
        received_channel_enc_output = net.channel.fading(channel_enc_output, 1, n_std)
    else:
        received_channel_enc_output = net.channel.fading(channel_enc_output, 0, n_std)

    for i in range(args.max_length):
        # create sequence padding
        look_ahead_mask = create_look_ahead_mask(outputs.size(1)).to('cuda')
        dec_target_padding_mask = create_padding_mask(outputs).to('cuda')
        combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

        # channel decoder
        received_channel_dec_output = net.channel_decoder(received_channel_enc_output)
        # semantic deocder
        predictions, _ = net.semantic_decoder(outputs, received_channel_dec_output,
                                                   False, combined_mask, enc_padding_mask)

        # choose the word from axis = 1
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = torch.argmax(predictions, dim=-1).long()

        outputs = torch.cat([outputs, predicted_id], dim=-1)

    return outputs


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(5)
    # choose performance metrics
    test_metrics = True
    test_bleu = True
    test_sentence_sim = False
    test_mi = False
    runs = 10
    SNR = [6]
    # Set Parameters
    args = helper()
    # Load the vocab
    vocab = json.load(open(args.vocab_path, 'r'))
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, args.end_idx)
    # Load dataset
    
    train_dataset, test_dataset = EuroparlDataset(args.train_path), EuroparlDataset(args.test_path)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
    

    # Load the model from the checkpoint path
    net = torch.load('/home/zhuzengle/multi_modal/DeepSC-Pytorch/checkpoint/epoch_0.pth')
    

    if test_metrics:
        if test_sentence_sim:
            metrics = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
        elif test_bleu:
            metrics = BleuScore(1, 0, 0, 0)
        else:
            raise Exception('Must choose bleu score or sentence similarity')
        # Start the evaluation
        # for snr in SNR:
        n_std = SNR_to_noise(args.test_snr)
        word, target_word = [], []
        score = 0
        for run in tqdm(range(runs)):
            for batch, (inp, tar) in enumerate(test_loader):
                inp, tar = inp.to('cuda'), tar.to('cuda')
                preds = greedy_decode(args, inp, net, args.channel, n_std)
                sentences = preds.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, sentences))
                word = word + result_string

                target_sent = tar.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, target_sent))
                target_word = target_word + result_string

            score1 = metrics.compute_score(word, target_word)
            score1 = np.array(score1)
            score1 = np.mean(score1)
            score += score1
            print(
                'Run: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                    run, score1
                )
            )
        score = score / runs
        print(
            'SNR: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                args.test_snr, score
            )
        )
