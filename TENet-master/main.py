import argparse
import math
import time
import datetime
import torch
import torch.nn as nn
from models import TENet,LSTNet,CNN,RNN,MHA_Net,MultiHeadAttention,GRU_attention,MTEGNN,rTEGNN
from MTGNN import gtnet,DataLoaderS
import numpy as np
import importlib
import sys

from utils import *
from ml_eval import *
import Optim
from torch.optim.lr_scheduler import LambdaLR
np.seterr(divide='ignore',invalid='ignore')
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    with torch.no_grad():
        for X, Y in data.get_batches(X, Y, batch_size, False):
            if X.shape[0]!=args.batch_size:
                break
            output = model(X)

            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))

            scale = data.scale.expand(output.size(0), data.m)
            # print('pred',output * scale)
            # print('y',Y * scale)
            #
            # print('n_sample',output.size(0) * data.m)
            # print('mse loss',evaluateL2(output * scale, Y * scale).item())
            # print('l1 loss', evaluateL1(output * scale, Y * scale).item())
            # print()
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)
            del scale,X,Y
            torch.cuda.empty_cache()

    rmse = math.sqrt(total_loss/n_samples)
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis = 0)
    sigma_g = (Ytest).std(axis = 0)
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g)
    correlation = (correlation[index]).mean ()
    mae = total_loss_l1/n_samples
    # if np.isnan(correlation):
    #     print(sigma_g)
    # (rse,rae,correlation) = (0,0,0)
    return rmse,rse, mae,rae,correlation
def evaluate1(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    with torch.no_grad():
        for X, Y in data.get_batches(X, Y, batch_size, False):
            if X.shape[0]!=args.batch_size:
                break
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            output = model(X)
            output = torch.squeeze(output)
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))

            scale = data.scale.expand(output.size(0), data.m)
            # print('pred',output * scale)
            # print('y',Y * scale)
            #
            # print('n_sample',output.size(0) * data.m)
            # print('mse loss',evaluateL2(output * scale, Y * scale).item())
            # print('l1 loss', evaluateL1(output * scale, Y * scale).item())
            # print()
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)
            del scale,X,Y
            torch.cuda.empty_cache()

    rmse = math.sqrt(total_loss/n_samples)
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis = 0)
    sigma_g = (Ytest).std(axis = 0)
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g)
    correlation = (correlation[index]).mean ()
    mae = total_loss_l1/n_samples
    # if np.isnan(correlation):
    #     print(sigma_g)
    # (rse,rae,correlation) = (0,0,0)
    return rmse,rse, mae,rae,correlation
def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # self.scheduler = optim.lr_criterionscheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1)  # 1/(epoch+1))
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad()
        # optimizer.zero_grad()
        id = [0, 1, 2, 3, 4, 5, 6, 7]
        id = torch.tensor(id).to(device)
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        # optimizer.step()
        grad_norm = optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.m)

        # del loss,output,scale,grad_norm
        torch.cuda.empty_cache()
    return total_loss / n_samples
def train1(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1)  # 1/(epoch+1))
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad()
        # optimizer.zero_grad()
        tx = X.unsqueeze(1).permute(0, 1, 3, 2)
        # perm = np.random.permutation(range(args.num_nodes))
        # id = perm[0:]
        id = [0,1,2,3,4,5,6,7]
        id = torch.tensor(id).to(device)
        tx = tx[:, :, id, :]
        ty = Y[:, id]

        output = model(tx,id)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, ty * scale)
        loss.backward()
        # optimizer.step()

        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
        grad_norm = optim.step()
        # del loss,output,scale,grad_norm
        # torch.cuda.empty_cache()
    return total_loss / n_samples
def classification_train(data,X,Y,model,criterion,optim,batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1)  # 1/(epoch+1))
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0] != args.batch_size:
            break
        model.zero_grad()
        # optimizer.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        # optimizer.step()
        grad_norm = optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.m)

        # del loss,output,scale,grad_norm
        torch.cuda.empty_cache()
    return total_loss / n_samples
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
# exchange_rate.txt energydata_complete.txt nasdaq100_padding.csv traffic.txt electricity.txt solar_AL.txt
parser.add_argument('--data', type=str, default="data/exchange_rate.txt",help='location of the data file')
parser.add_argument('--n_e', type=int, default=8,help='The number of graph nodes')
parser.add_argument('--model', type=str, default='rTEGNN',help='')
parser.add_argument('--k_size', type=list, default=[3,5,7],help='number of CNN kernel sizes')
parser.add_argument('--window', type=int, default=32,help='window size')
parser.add_argument('--decoder', type=str, default= 'rGNN',help = 'type of decoder layer')
parser.add_argument('--horizon', type=int, default= 3)
parser.add_argument('--num_adj', type=int, default= 3)
parser.add_argument('--A', type=str, default="TE/exte.txt",help='A')
parser.add_argument('--B', type=str, default="TE/ex_corr.txt",help='B')
parser.add_argument('--highway_window', type=int, default=0
                    , help='The window size of the highway component')
parser.add_argument('--epochs', type=int, default=100,help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',help='batch size')


parser.add_argument('--skip_mode', type=str, default="none",help='skipmode')
parser.add_argument('--attention_mode', type=str, default="naive",help='attention_mode')
parser.add_argument('--channel_size', type=int, default=12,help='the channel size of the CNN layers')
parser.add_argument('--hid1', type=int, default=40,help='the hidden size of the GNN layers')
parser.add_argument('--hid2', type=int, default=10,help='the hidden size of the GNN layers')

parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units each layer')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')

parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units (channels)')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')


parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)


# parser.add_argument('--clip', type=float, default=10,help='gradient clipping')


# parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)


parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=10)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='Linear')

# MTGNN args
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nod'
                    'es',type=int,default=8,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=4,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=12,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=12,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=32,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=10,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')
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

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)

device = torch.device(args.device)
#MTGNN model
if args.model == 'MTEGNN':
    model = eval(args.model).Model(args,args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                      device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                      node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                      conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                      skip_channels=args.skip_channels, end_channels= args.end_channels,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
else:
    model = eval(args.model).Model(args,Data)
#
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average = False).cuda()
else:
    criterion = nn.MSELoss(size_average = False).cuda()
evaluateL2 = nn.MSELoss(size_average = False).cuda()
evaluateL1 = nn.L1Loss(size_average = False).cuda()
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()
    
    
best_val = 111110
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,lr_decay=args.weight_decay
)

# print('begin var')
# test_mse,test_acc, test_mae,test_rae, test_corr = evaluate_VAR(args,evaluateL2,evaluateL1)
# print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_mse,test_acc, test_mae,test_rae, test_corr))

ttime = str(datetime.datetime.now()).replace(' ','-')
save_model = args.save+ttime
try:
    print('begin training')
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        train_time = time.time()-epoch_start_time
        val_rmse,val_rse, val_mae,val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)

        print('| end of epoch {:3d} | time: {:5.4f}s | train_loss {:5.5f} | valid rmse {:5.5f} |valid rse {:5.5f} | valid mae {:5.5f} | valid rae {:5.5f} |valid corr  {:5.5f}'.format(epoch, train_time, train_loss, val_rmse,val_rse, val_mae,val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.
        # if str(val_corr) == 'nan':
        #     sys.exit()
        # if float(val_corr)<0.4:
        #     sys.exit()
        val = val_mae
        if args.decoder == 'GIN':
            val = val_rse

        if val < best_val:
            with open(save_model, 'wb') as f:
                torch.save(model, f)
            best_val = val
            # epoch_test_time = time.time()
            test_rmse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))
            # print(time.time()-epoch_test_time)
        else:
            test_rmse, test_acc, test_mae, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model,
                                                                          evaluateL2, evaluateL1, args.batch_size)
            print("\n          test rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(
                test_rmse, test_acc, test_mae, test_rae, test_corr))

        # if epoch % 5 == 0:
        #     test_rmse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
        #     print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(save_model, 'rb') as f:
    model = torch.load(f)
test_mse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_mse,test_acc, test_mae,test_rae, test_corr))

with open('results.txt','a+') as f1:
    f1.write(str(args.model)+' '+str(args.data) + ' '+str(args.horizon) + ' '+str(args.num_adj)+' '+str(args.channel_size)+' '+str(args.hid1)+' '+str(args.hid2)+ ' '+str(args.highway_window)+' '+str(args.window)+' '+save_model+'\n'+str(test_mse)+' '+str(test_acc)+ ' '+str(test_mae)+' '+str(test_rae)+' '+str(test_corr))
    f1.write('\n')

# time.sleep(300)