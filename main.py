'''
Author: LOTEAT
Date: 2023-05-31 15:56:52
'''
import torch
from helper import helper
import json
from utils import snr2noise
from dataset import EuroparlDataset
from torch.utils.data import DataLoader
from models.transceiver import Transceiver
from models.mine import Mine
import torch.optim as optim
import torch
from train import train
from tqdm import tqdm
import os

def load_vocab(args):
    vocab = json.load(open(args.vocab_path, 'rb'))
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"] 
    print(args.vocab_size)
    return args


if __name__ == '__main__':
    # Set random seed
    # torch.manual_seed(5)
    seed = 123
    torch.manual_seed(seed)

    # 如果使用GPU，还需设置以下两行代码
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    args = helper()
    torch.set_num_threads(args.nthreads)
    os.makedirs(args.save_path, exist_ok=True)
    
    # Load the vocab
    args = load_vocab(args)
    
    # Load dataset
    train_dataset, test_dataset = EuroparlDataset(args.train_path), EuroparlDataset(args.test_path)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    transeiver = Transceiver(args).to(device)
    mine = Mine().to(device)     

    # Define the optimizer
    transeiver_optim = optim.Adam(transeiver.parameters(), lr=args.trans_lr)
    mine_optim = optim.Adam(mine.parameters(), lr=args.mine_lr)
    
    # Training the model
    best_loss = 10
    for epoch in tqdm(range(args.epochs)):
        n_std = snr2noise(args.train_snr)
        train_loss_record, test_loss_record = 0, 0
        for (batch, (data, target)) in enumerate(train_loader):
            
            data, target = data.to(device), target.to(device)
            train_loss, train_loss_mine, _ = train(data, target, transeiver, mine, transeiver_optim, mine_optim, args.channel, n_std,
                                            use_mine=args.use_mine)
            train_loss_record += train_loss
        torch.save(transeiver, "%s/epoch_%d.pth" % (args.save_path, epoch))
        train_loss_record = train_loss_record/batch
        

        # for (batch, (inp, tar)) in enumerate(test_dataset):
        #     test_loss = eval_step(inp, tar, net, args.channel, n_std)
        #     test_loss_record += test_loss
        # test_loss_record = test_loss_record / batch

        # if best_loss > test_loss_record:
        #     best_loss = test_loss_record
        #     manager.save(checkpoint_number=epoch)

        # print('Epoch {} Train Loss {:.4f} Test Loss {:.4f}'.format(epoch + 1, train_loss_record, test_loss_record))



# {
#     // 使用 IntelliSense 了解相关属性。 
#     // 悬停以查看现有属性的描述。
#     // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python: 当前文件",
#             "type": "python",
#             "request": "launch",
#             "program": "run_sc.py",
#             "console": "integratedTerminal",
#             "justMyCode": false,
#             "args": ["--test_only", "--load_from", "epoch_1.pth"]
#         }
#     ]
# }