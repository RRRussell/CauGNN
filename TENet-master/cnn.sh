#!/usr/bin/env bash

# Best rGNN for energy
#python main.py --hid1 40 --hid2 10 --data data/energydata_complete.txt --n_e 26 --A TE/ente.txt --B TE/energy_corr.txt --subgraph_size 8

# Best rGNN for exchange_rate
#python main.py --hid1 20 --hid2 10 --highway_window 0


python main.py --horizon 3 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt --model LSTNet --window 168 --highway_window 24 --epochs 30
python main.py --horizon 6 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt --model LSTNet --window 168 --highway_window 24 --epochs 30
python main.py --horizon 12 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt --model LSTNet --window 168 --highway_window 24 --epochs 30
python main.py --horizon 24 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt --model LSTNet --window 168 --highway_window 24 --epochs 30

python main.py --horizon 3 --data data/energydata_complete.txt --n_e 26 --model RNN --window 168 --highway_window 24 --epochs 50
python main.py --horizon 6 --data data/energydata_complete.txt --n_e 26 --model RNN --window 168 --highway_window 24 --epochs 50
python main.py --horizon 12 --data data/energydata_complete.txt --n_e 26 --model RNN --window 168 --highway_window 24 --epochs 50
python main.py --horizon 24 --data data/energydata_complete.txt --n_e 26 --model RNN --window 168 --highway_window 24 --epochs 50

python main.py --horizon 3 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt
python main.py --horizon 6 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt
python main.py --horizon 12 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt
python main.py --horizon 24 --data data/nasdaq100_padding.csv --n_e 82 --A TE/nate.txt --B TE/nasdaq_corr.txt