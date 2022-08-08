import numpy as np
raw_file = '/home/jiangnanyida/Downloads/classification_data/occupancy_data/datatest.txt'
target_loc = '/home/jiangnanyida/Downloads/Coffee/dataset'
# f = np.loadtxt(raw_file,dtype =<class 'string'>,delimiter=',')
a = np.genfromtxt(raw_file, delimiter=',', skip_header=True)[:, 2:]
print(a.shape)