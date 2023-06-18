'''
Author: LOTEAT
Date: 2023-06-18 15:58:33
'''
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return words


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_score(self, real, predicted):
        score1 = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score1.append(sentence_bleu([sent1], sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
        return score1


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


class Similarity():
    def __init__(self, config_path, checkpoint_path, dict_path):
        self.model1 = BertModel.from_pretrained(checkpoint_path)
        self.model = nn.Sequential(*list(self.model1.children())[:-1])
        self.tokenizer = BertTokenizer.from_pretrained(dict_path, do_lower_case=True)

    def compute_score(self, real, predicted):
        token_ids1, segment_ids1 = [], []
        token_ids2, segment_ids2 = [], []
        score = []

        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            inputs1 = self.tokenizer.encode_plus(sent1, add_special_tokens=True, max_length=32, truncation=True)
            inputs2 = self.tokenizer.encode_plus(sent2, add_special_tokens=True, max_length=32, truncation=True)

            token_ids1.append(inputs1['input_ids'])
            token_ids2.append(inputs2['input_ids'])
            segment_ids1.append(inputs1['token_type_ids'])
            segment_ids2.append(inputs2['token_type_ids'])

        token_ids1 = torch.tensor(token_ids1)
        token_ids2 = torch.tensor(token_ids2)
        segment_ids1 = torch.tensor(segment_ids1)
        segment_ids2 = torch.tensor(segment_ids2)

        vector1 = self.model(token_ids1, segment_ids1)[0]
        vector2 = self.model(token_ids2, segment_ids2)[0]

        vector1 = torch.sum(vector1, dim=1)
        vector2 = torch.sum(vector2, dim=1)

        vector1 = normalize(vector1.detach().numpy(), axis=0, norm='max')
        vector2 = normalize(vector2.detach().numpy(), axis=0, norm='max')

        dot = np.diag(np.matmul(vector1, vector2.T))
        a = np.diag(np.matmul(vector1, vector1.T))
        b = np.diag(np.matmul(vector2, vector2.T))

        a = np.sqrt(a)
        b = np.sqrt(b)

        output = dot / (a * b)
        score = output.tolist()

        return score
