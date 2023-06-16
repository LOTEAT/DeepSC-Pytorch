'''
Author: LOTEAT
Date: 2023-05-31 18:37:44
'''
import torch.nn as nn
from .semantic.encoder import SemanticEncoder
from .semantic.decoder import SemanticDecoder
from .channel.encoder import ChannelEncoder
from .channel.decoder import  ChannelDecoder
from .channel.channel import Channel

class Transceiver(nn.Module):
    def __init__(self, args):
        super(Transceiver, self).__init__()

        # semantic encoder
        self.semantic_encoder = SemanticEncoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        # semantic decoder
        self.semantic_decoder = SemanticDecoder(args.decoder_num_layer, args.decoder_num_heads,
                                        args.decoder_d_model, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)

        # channel encoder
        self.channel_encoder = ChannelEncoder(256, 16)
        # channel decoder
        self.channel_decoder = ChannelDecoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channel()

    def forward(self, inputs, tar_inp, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):

        sema_enc_output = self.semantic_encoder(inputs, enc_padding_mask)
        # channel encoder
        channel_enc_output = self.channel_encoder(sema_enc_output)
        # over the AWGN channel
        if channel=='AWGN':
            # self.channel_layer.awgn(channel_enc_output, n_std)
            received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)
        elif channel=='Rician':
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 1, n_std)
        else:
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 0, n_std)

        # channel decoder
        received_channel_dec_output = self.channel_decoder(received_channel_enc_output)
        # semantic deocder
        predictions, _ = self.semantic_decoder(tar_inp, received_channel_dec_output,
                                                    training, combined_mask, dec_padding_mask)

        return predictions, channel_enc_output, received_channel_enc_output


    def train_semcodec(self, inputs, tar_inp, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):
        sema_enc_output = self.semantic_encoder(inputs, training, enc_padding_mask)
        predictions, _ = self.semantic_decoder(tar_inp, sema_enc_output,
                                                    training, combined_mask, dec_padding_mask)
        return predictions

    def train_chancodec(self, sema_enc_output, channel='AWGN', n_std=0.1):
        # channel encoder
        channel_enc_output = self.channel_encoder(sema_enc_output)
        # over the air
        if channel == 'AWGN':
            received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)
        elif channel == 'Rician':
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 1, n_std)
        else:
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 0, n_std)
        # channel decoder
        received_channel_dec_output = self.channel_decoder(received_channel_enc_output)

        return received_channel_dec_output