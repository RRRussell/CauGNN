#!/usr/bin/env bash
python main.py --model RNN --data data/exchange_rate.txt --hidCNN 50 --hidRNN 10 --L1Loss False --output_fun None
