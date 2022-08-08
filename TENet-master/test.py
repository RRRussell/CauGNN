import numpy as np
# A = np.eye(137)
# with open('TE/solar.txt','r') as f:
#     for line in f.readlines():
#         if not line:
#             continue
#         i,b = line.split('-')
#         print(i,b)
#         j,v = b.split(':')
#         A[int(i)][int(j)] = float(v)
# print(A)
# f1 = open('TE/so.txt','w')
# for i in range(137):
#     for j in range(137):
#         f1.write(str(A[i,j])+' ')
#     f1.write('\n')
# f1.close()
file = 'data/exchange_rate.txt'
A = np.loadtxt(file, delimiter=',').T
f1 = open('ex_T.txt','w')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        f1.write(str(A[i,j])+',')
    f1.write('\n')
f1.close()
print(A.shape)
