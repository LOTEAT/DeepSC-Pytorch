import torch
import torch.nn as nn
from .semantic import SemanticEncoder, SemanticDecoder
from .channel import ChannelEncoder, ChannelDecoder, Channel


class Mine(nn.Module):
    def __init__(self, hidden_size=10):
        super(Mine, self).__init__()
        randN_05 = torch.nn.init.Normal(mean=0.0, std=0.02)
        bias_init = torch.nn.init.Constant(0)

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense1.weight.data = randN_05(self.dense1.weight.data)
        self.dense1.bias.data = bias_init(self.dense1.bias.data)
        self.relu1 = nn.ReLU()

        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense2.weight.data = randN_05(self.dense2.weight.data)
        self.dense2.bias.data = bias_init(self.dense2.bias.data)
        self.relu2 = nn.ReLU()

        self.dense3 = nn.Linear(hidden_size, 1)
        self.dense3.weight.data = randN_05(self.dense3.weight.data)
        self.dense3.bias.data = bias_init(self.dense3.bias.data)

    def forward(self, inputs):
        output1 = self.dense1(inputs)
        output1 = self.relu1(output1)
        output2 = self.dense2(output1)
        output2 = self.relu2(output2)
        output = self.dense3(output2)
        return output


class Transeiver(nn.Module):
    def __init__(self, args):
        super(Transeiver, self).__init__()

        # semantic encoder
        self.semantic_encoder = SemanticEncoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        # semantic decoder
        self.semantic_decoder = SemanticDecoder(args.decoder_num_layer, args.decoder_d_model,
                                        args.decoder_num_heads, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)

        # channel encoder
        self.channel_encoder = ChannelEncoder(256, 16)
        # channel decoder
        self.channel_decoder = ChannelDecoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channel()

    def forward(self, inputs, tar_inp, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):

        sema_enc_output = self.semantic_encoder(inputs, training, enc_padding_mask)
        # channel encoder
        channel_enc_output = self.channel_encoder(sema_enc_output)
        # over the AWGN channel
        if channel=='AWGN':
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