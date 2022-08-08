# TEGNN

This repository is the official implementation of [Multivariate Time Series Forecasting with Transfer Entropy Graph](https://ieeexplore.ieee.org/document/9837007). 

## Requirements
- python 3.7.7
- Pytorch 1.4.0

To install requirements:

```setup
pip install -r requirements.txt
```

## Overview

#### Dataset

We conduct experiments on three benchmark datasets for multivariate time series forecasting tasks, this table shows dataset statistics

| Dataset      | T                  | D       | L          |
| -------------|--------------------| --------| -----------|
| Exchange_rate|7588                |    8    |  1 day     |
|              |                    |         |            |
| Energy       |19736               |    26   |  10 minutes|
|              |                    |         |            |
| Nasdaq       |40560               |    82   |  1 minute  |
|              |                    |         |            |

where `T` is the length of time series, `D` is the number of variables, `L` is the sample rate.
 
#### Preprocessing
We split the raw data into `train set`, `validation set` and `test set`, in the ratio of 6:2:2. 

In each set, consecutive time series
with certain length of `window size` are sampled as a slice, which forms a forecasting unit. The window size is set to 32 for TEGNN model. The slice window moves over the entire 
time series in the pace of 1 step each time.  

#### Transfer Entropy matrix

We use R to measure information of transfer between different time series. To get the Transfer Entropy matrix, run this code under `data` folder:
```TE matrix
Rscript rte.R
```
and place the result files under `TE` folder.

We also implement a python version of Transfer Entropy measurement, run this code:
```TE matrix 2
python Teoriginal.py
```

## Training

To train the model(s) in the paper, run this code:

```train
python train.py --model TENet --channel_size 12 --hid1 30 --hid2 10
```

## Evaluation

To evaluate the model in the paper, run this code:

```eval
python eval.py --model_file model.pt --data data/exchange_rate.txt --horizon 5
```

## Pre-trained Models

You can download pre-trained models here:

- [TEGNN pre-trained model](https://drive.google.com/drive/folders/18y6ud7-uOyDPDaUYnzkCHGnKACSRdyOV?usp=sharing) trained on different datasets. 

and place the pre-trained model under `model` folder. Note that this model should be loaded directly with Pytorch,
or passed to `eval.py`.

On a Tesla V100, it took 0.20 seconds per epoch on Exchange_rate dataset, in the default setting of hyper parameters.



## Results

We train TEGNN for 1000 epochs for each train option, and use the model that has the best performance on validation
set for test. 
 
We use three conventional evaluation metrics to evaluate the performance of TEGNN model: Mean Absolute Error(**MAE**),
Relative Absolute Error(**RAE**) and Empirical Correlation Coefficient(**CORR**), the following table shows the results:



| Model name| Dataset            | horizon | MAE    | RAE    | CORR   |
| ----------|--------------------| --------| -------| -------| -------|
| ----------|--------------------| --------| -------| -------| -------|
|           |                    |    5    |  0.0060| 0.0176 | 0.9694 |
| TEGNN     |exchange_rate       |    10   |  0.0083| 0.0243 | 0.9548 |
|           |                    |    15   |  0.0104| 0.0302 | 0.9438 |
| ----------|--------------------| --------| -------| -------| -------|
|           |                    |    5    |  2.0454| 0.0358 | 0.9267 |
| TEGNN     |energydata_complete |    10   |  2.7242| 0.0470 | 0.8673 |
|           |                    |    15   |  3.3232| 0.0573 | 0.8221 |
| ----------|--------------------| --------| -------| -------| -------|
|           |                    |    5    |  0.1549| 0.0010 | 0.9951 |
| TEGNN     |nasdaq              |    10   |  0.1897| 0.0012 | 0.9922 |
|           |                    |    15   |  0.0015| 0.0302 | 0.9887 |

Examples with parameters to run different datasets are in `runExchangeRate.sh`, `runEnergy.sh` and `runNasdaq.sh`, in which specific
hyperparameters for each training options are listed.






