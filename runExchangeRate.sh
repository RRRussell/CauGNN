#!/usr/bin/env bash

python train.py --model TENet --window 32 --horizon 5 --highway_window 1 --channel_size 12 --hid1 40 --hid2 10 --data data/exchange_rate.txt --n_e 8 --A TE/exte.txt
python train.py --model TENet --window 32 --horizon 10 --highway_window 1 --channel_size 12 --hid1 40 --hid2 10 --data data/exchange_rate.txt --n_e 8 --A TE/exte.txt
python train.py --model TENet --window 32 --horizon 15 --highway_window 1 --channel_size 12 --hid1 40 --hid2 10 --data data/exchange_rate.txt --n_e 8 --A TE/exte.txt







