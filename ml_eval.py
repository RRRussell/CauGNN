from statsmodels.tsa.seasonal import seasonal_decompose
import argparse
import math
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import numpy as np
from math import sqrt
import torch.nn as nn
# VAR example
from utils import *
import torch
from random import random
# contrived dataset with dependency

def evaluate_VAR(args,evaluateL2, evaluateL1):
    Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    ti = 0
    for X, Y in Data.get_batches(Data.test[0], Data.test[1], args.batch_size, False):
        if X.shape[0]!=128:
            break

        output = torch.zeros([args.batch_size,args.n_e])
        print(ti)
        ti += 1
        for i in range(X.shape[0]):
            x_t = X.permute(0,2,1)[i].cpu().numpy().T
            d = list()
            for row in x_t:
                # print(row)
                for l in range(len(row)):
                    row[l] += random()/10000000

                d.append(row)
            # print(d)
            model = VAR(d)
            model_fit = model.fit()
            # make prediction
            yhat = model_fit.forecast(model_fit.y, steps=args.horizon)[args.horizon-1]
            output[i] = torch.from_numpy(yhat).view(args.n_e)
            # print('\n\n')
            # print(model_fit.forecast(model_fit.y, steps=args.horizon))
            # print(Y[i])
        # print(output)
        output = output.cuda()

        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict,output))
            test = torch.cat((test, Y))

        scale = Data.scale.expand(output.size(0), args.n_e)
        # print('pred',output * scale)
        # print('y',Y * scale)
        #
        # print('n_sample',output.size(0) * data.m)
        # print('mse loss',evaluateL2(output * scale, Y * scale).item())
        # print('l1 loss', evaluateL1(output * scale, Y * scale).item())
        # print()
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * args.n_e)
        del scale,X,Y
        torch.cuda.empty_cache()

    mse = math.sqrt(total_loss/n_samples)
    rse = math.sqrt(total_loss / n_samples)/Data.rse
    rae = (total_loss_l1/n_samples)/Data.rae
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
    return mse,rse, mae,rae,correlation

def evaluate_AR(args,evaluateL2, evaluateL1):
    Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    ti = 0
    for X, Y in Data.get_batches(Data.test[0], Data.test[1], args.batch_size, False):
        if X.shape[0]!=128:
            break
        output = torch.zeros([args.batch_size,args.n_e])
        print(ti)
        ti += 1
        for i in range(X.shape[0]):
            x_t = X.permute(0,2,1)[i].cpu().numpy()
            y = list()
            for row in x_t:
                d = list(row)
                # print(d)
                # print(d)
                model = AutoReg(d, lags=5)
                model_fit = model.fit()
                # make prediction
                yhat = model_fit.predict(len(d),len(d)+args.horizon)[args.horizon-1]
                y.append(yhat)
                # print(yhat)
            output[i] = torch.tensor(y)
            # print(y)
            # print(Y[i])
            # print('\n\n')
            # print(model_fit.forecast(model_fit.y, steps=args.horizon))
            # print(Y[i])
        # print(output)
        output = output.cuda()

        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict,output))
            test = torch.cat((test, Y))

        scale = Data.scale.expand(output.size(0), args.n_e)
        # print('pred',output * scale)
        # print('y',Y * scale)
        #
        # print('n_sample',output.size(0) * data.m)
        # print('mse loss',evaluateL2(output * scale, Y * scale).item())
        # print('l1 loss', evaluateL1(output * scale, Y * scale).item())
        # print()
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * args.n_e)
        del scale,X,Y
        torch.cuda.empty_cache()

    mse = math.sqrt(total_loss/n_samples)
    rse = math.sqrt(total_loss / n_samples)/Data.rse
    rae = (total_loss_l1/n_samples)/Data.rae
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
    return mse,rse, mae,rae,correlation