'''
Author: LOTEAT
Date: 2023-05-31 19:47:27
'''
import torch
import torch.nn as nn

class SparseCategoricalCrossentropyLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(SparseCategoricalCrossentropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_object = nn.CrossEntropyLoss(reduction='none')

    def forward(self, real, pred):
        mask = real != self.ignore_index
        bs = pred.shape[0]

        loss_ = self.loss_object(pred.contiguous().view(1500, 22234), real.contiguous().view(-1))#  * mask.float()
        loss_ = loss_.view(bs, -1)
        loss_ *= mask.float()

        return torch.mean(loss_)


