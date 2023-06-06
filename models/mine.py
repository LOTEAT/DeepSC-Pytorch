'''
Author: LOTEAT
Date: 2023-06-06 19:51:39
'''
import torch
import torch.nn as nn

class Mine(nn.Module):
    def __init__(self, hidden_size=10):
        super(Mine, self).__init__()
        normal_init = lambda x: torch.nn.init.normal(x, mean=0.0, std=0.02)
        zero_init = lambda x: torch.nn.init.constant(x, 0)

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(hidden_size, 1)
        
        for dense in [self.dense1, self.dense2, self.dense3]:
            normal_init(dense.weight.data)
            zero_init(dense.bias.data) 
        

    def forward(self, inputs):
        output1 = self.dense1(inputs)
        output1 = self.relu1(output1)
        output2 = self.dense2(output1)
        output2 = self.relu2(output2)
        output = self.dense3(output2)
        return output