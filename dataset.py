'''
Author: LOTEAT
Date: 2023-05-31 16:34:26
'''
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class EuroparlDataset(Dataset):
    def __init__(self, path, length=-1):
        data = pickle.load(open(path, 'rb'))
        data = data[:length]
        self.data = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in data], batch_first=True)

    def __getitem__(self, index):
        return torch.LongTensor(self.data[index]), torch.LongTensor(self.data[index])

    def __len__(self):
        return len(self.data)

