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
        self.semantic_encoder = SemanticEncoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        self.semantic_decoder = SemanticDecoder(args.decoder_num_layer, args.decoder_num_heads,
                                        args.decoder_d_model, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)

        self.channel_encoder = ChannelEncoder(args.encoder_d_model)
        self.channel_decoder = ChannelDecoder(args.decoder_d_model)

        self.channel = Channel()

    def forward(self, data, target, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):

        channel_enc_out = self.channel_encoder(self.semantic_encoder(data, enc_padding_mask))

        if channel=='AWGN':
            channel_out = self.channel.awgn(channel_enc_out, n_std)
        elif channel=='Rician':
            channel_out = self.channel.fading(channel_enc_out, 1, n_std)
        else:
            channel_out = self.channel.fading(channel_enc_out, 0, n_std)

        # semantic deocder
        pred, _ = self.semantic_decoder(target, self.channel_decoder(channel_out),
                                                    training, combined_mask, dec_padding_mask)

        return pred, channel_enc_out, channel_out


    def train_semcodec(self, inputs, tar_inp, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):
        sema_enc_output = self.semantic_encoder(inputs, training, enc_padding_mask)
        predictions, _ = self.semantic_decoder(tar_inp, sema_enc_output,
                                                    training, combined_mask, dec_padding_mask)
        return predictions

    def train_chancodec(self, sema_enc_output, channel='AWGN', n_std=0.1):
        channel_enc_out = self.channel_encoder(sema_enc_output)
        if channel == 'AWGN':
            channel_out = self.channel.awgn(channel_enc_out, n_std)
        elif channel == 'Rician':
            channel_out = self.channel.fading(channel_enc_out, 1, n_std)
        else:
            channel_out = self.channel.fading(channel_enc_out, 0, n_std)
        channel_dec_out = self.channel_decoder(channel_out)
        return channel_dec_out