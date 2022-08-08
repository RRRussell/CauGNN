import argparse
import math
import time

import torch
import torch.nn as nn
from models import TENet, LSTNet, CNN, RNN, MHA_Net, MultiHeadAttention, GRU_attention
import numpy as np
import importlib

from utils import *
from ml_eval import *
import Optim
from torch.optim.lr_scheduler import LambdaLR

np.seterr(divide='ignore', invalid='ignore')


def evaluate(data, X, Y, model, evaluateL2, evaluateL1):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    with torch.no_grad():
        X,Y = data.get_batches(X, Y)
        output = model(X)

        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()

    return total_loss

def classification_train(data, X, Y, model, optim):
    model.train()
    total_loss = 0
    n_samples = 0
    X, Y  = data.get_batches(X, Y, True)
    model.zero_grad()
    # optimizer.zero_grad()
    output = model(X)
    loss = model.get_loss(output,labels=Y,idx_lst=None)
    loss.backward()
    grad_norm = optim.step()
    total_loss += loss.data.item()
    n_samples += output.size(0)

    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
# exchange_rate.txt energydata_complete.txt nasdaq100_padding.csv
parser.add_argument('--train_data', type=str, default="data/Adiac_TRAIN.txt", help='location of the train data file')
parser.add_argument('--test_data', type=str, default="data/Adiac_TEST.txt", help='location of the test data file')
parser.add_argument('--n_e', type=int, default=8, help='The number of graph nodes')
parser.add_argument('--model', type=str, default='HGNN', help='')
parser.add_argument('--k_size', type=list, default=[3, 5, 7], help='number of CNN kernel sizes')
parser.add_argument('--window', type=int, default=32, help='window size')
parser.add_argument('--decoder', type=str, default='GNN', help='type of decoder layer')
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--num_adj', type=int, default=2)
# parser.add_argument('--A', type=str, default="TE/exte.txt", help='A')
# parser.add_argument('--B', type=str, default="TE/ex_corr.txt", help='B')
parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')
parser.add_argument('--attention_mode', type=str, default="naive", help='attention_mode')

parser.add_argument('--channel_size', type=int, default=12, help='the channel size of the CNN layers')
parser.add_argument('--hid1', type=int, default=256, help='the hidden size of the GNN layers')
parser.add_argument('--hid2', type=int, default=8, help='the hidden size of the GNN layers')

parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')

parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321, help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--L1Loss', type=bool, default=True)
args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = STS_Data_utility(args.train_file,args.test_file, args.cuda)

model = eval(args.model).Model(args, Data)
#
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False).cpu()
else:
    criterion = nn.MSELoss(size_average=False).cpu()
evaluateL2 = nn.MSELoss(size_average=False).cpu()
evaluateL1 = nn.L1Loss(size_average=False).cpu()
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

best_val = 111110
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

try:
    print('begin training')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = classification_train(Data, Data.x_train, Data.y_train, model, optim)

        val_loss = evaluate(Data, Data.x_test, Data.y_test, model, evaluateL2,evaluateL1)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | val_loss {:5.5f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
        # Save the model if the validation loss is the best we've seen so far.
        val = val_loss

        if val < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

