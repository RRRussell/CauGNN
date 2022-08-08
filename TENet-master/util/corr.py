import pandas as pd
import numpy as np

option = 2
if option == 1:
    file = '../data/nasdaq100_padding.csv'
    A = np.loadtxt(file,delimiter=',')
    A = np.array(A,dtype=np.float32).T
    # print(A[0])
    d = {}
    for i in range(A.shape[0]):
        d[i] = A[i]

    # df = pd.DataFrame({
    #     'a': [11, 22, 33, 44, 55, 66, 77, 88, 99],
    #     'b': [10, 24, 30, 48, 50, 72, 70, 96, 90],
    #     'c': [91, 79, 72, 58, 53, 47, 34, 16, 10],
    #     'd': [99, 10, 98, 10, 17, 10, 77, 89, 10]})
    df = pd.DataFrame(d)
    df_corr = df.corr()
    # 可视化
    import matplotlib.pyplot as mp
    import seaborn
    seaborn.heatmap(df_corr, center=0, annot=True)
    # mp.show()
    X = df_corr.to_numpy()

    f1 = open('../TE/nasdaq_corr.txt','w')
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            f1.write(str(X[i][j])+' ')
        f1.write('\n')
    f1.close()

file = '/home/jiangnanyida/Documents/MTS/MTS_TEGNN/TENet-master/data/tep7.txt'
A = np.loadtxt(file)
B = A.T
print(B[0][0])
print(B[1][0])
print(B[2][0])
print(B.shape)
f1 = open('/home/jiangnanyida/Documents/MTS/MTS_TEGNN/TENet-master/data/tep7_refine.txt','w')
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        f1.write(str(B[i][j])+' ')
    f1.write('\n')
f1.close()