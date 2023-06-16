'''
Author: LOTEAT
Date: 2023-05-31 15:59:28
'''

import argparse


def helper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nthreads', default=3, type=int)
    # preprocessing parameters
    parser.add_argument('--in_dir', default='europarl/en', type=str)
    parser.add_argument('--out_train_dir', default='europarl/train_data.pkl', type=str)
    parser.add_argument('--out_test_dir', default='europarl/test_data.pkl', type=str)
    parser.add_argument('--out_vocab', default='europarl/vocab.json', type=str)

    parser.add_argument('--train-path', default='data/europarl/train_data.pkl', type=str)
    parser.add_argument('--test-path', default='data/europarl/test_data.pkl', type=str)
    parser.add_argument('--vocab_path', default='data/europarl/vocab.json', type=str)
    parser.add_argument('--trans_lr', default=5e-4, type=float, help='The training learning rate')
    parser.add_argument('--mine_lr', default=1e-2, type=float, help='The training learning rate')


    # Training parameters
    parser.add_argument('--bs', default=64, type=int, help='The training batch size')
    parser.add_argument('--shuffle-size', default=2000, type=int, help='The training shuffle size')
    
    
    
    
    
    parser.add_argument('--epochs', default=60, type=int, help='The training number of epochs')
    parser.add_argument('--train-with-mine',  action='store_true',
                    help='If added, the network will be trained WITH Mutual Information')
    parser.add_argument('--checkpoint-path', default='./checkpoint', type=str,
                        help='The path to save model')
    parser.add_argument('--max-length', default=35, type=int, help='The path to save model')
    parser.add_argument('--channel', default='AWGN', type=str, help='Choose the channel to simulate')

    # Model parameters
    parser.add_argument('--encoder-num-layer', default=4, type=int, help='The number of encoder layers')
    parser.add_argument('--encoder-d-model', default=128, type=int, help='The output dimension of attention')
    parser.add_argument('--encoder-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--encoder-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--encoder-dropout', default=0.1, type=float, help='The encoder dropout rate')

    parser.add_argument('--decoder-num-layer', default=4, type=int, help='The number of decoder layers')
    parser.add_argument('--decoder-d-model', default=128, type=int, help='The output dimension of decoder')
    parser.add_argument('--decoder-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--decoder-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--decoder-dropout', default=0.1, type=float, help='The decoder dropout rate')


    # Other parameter settings
    parser.add_argument('--train-snr', default=3, type=int, help='The train SNR')
    parser.add_argument('--test-snr', default=6, type=int, help='The test SNR')
    # Mutual Information Model Parameters


    args = parser.parse_args()

    return args

