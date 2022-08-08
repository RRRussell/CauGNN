# -*- coding: utf-8 -*-
import numpy as np
from numba import jit
@jit
def TE(x,y,pieces,j):
    d_x=np.zeros((j,4))
    sit=len(x)
    temp1=list(range(sit-1))
    np.random.shuffle(temp1)
    select=np.array(temp1[:j])
    
    d_x[:,0]=x[select+1]
    d_x[:,1]=x[select]
    d_x[:,2]=y[select+1]
    d_x[:,3]=y[select]
    
    x_max=np.max(x); x_min=np.min(x); y_max=np.max(y); y_min=np.min(y)
    
    delta1=(x_max-x_min)/(2*pieces); delta2=(y_max-y_min)/(2*pieces)
    
    L1=np.linspace(x_min+delta1,x_max-delta1,pieces); L2=np.linspace(y_min+delta2,y_max-delta2,pieces)
    
    dist1=np.zeros((pieces,2))
    count=-1
    
    for q1 in range(pieces):
        k1=L1[q1]; k2=L2[q1]
        count+=1
        count1=0;count2=0
        for i in range(j):
            if d_x[i,1]>=(k1-delta1) and d_x[i,1]<=(k1+delta1):
                count1+=1
            if d_x[i,3]>=(k2-delta2) and d_x[i,3]<=(k2+delta2):
                count2+=1
        #print(dist1)
        #print(count1)
        dist1[count,0]=count1; dist1[count,1]=count2
        
    dist1[:,0]=dist1[:,0]/sum(dist1[:,0]); dist1[:,1]=dist1[:,1]/sum(dist1[:,1])
    
    dist2=np.zeros((pieces,pieces,3))
    for q1 in range(pieces):
        for q2 in range(pieces):
            # print('222')
            k1=L1[q1]; k2=L1[q2]
            k3=L2[q1]; k4=L2[q2]
            count1=0;count2=0;count3=0
            for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1):
                        count1=count1+1;
                
                    if d_x[i1,2]>=(k3-delta2) and d_x[i1,2]<=(k3+delta2) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2):
                        count2=count2+1;
                    
                    if d_x[i1,1]>=(k1-delta1) and d_x[i1,1]<=(k1+delta1) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2):
                        count3=count3+1;
                           
            dist2[q1,q2,0]=count1; dist2[q1,q2,1]=count2; dist2[q1,q2,2]=count3;
            
    dist2[:,:,0]=dist2[:,:,0]/np.sum(dist2[:,:,0])
    dist2[:,:,1]=dist2[:,:,1]/np.sum(dist2[:,:,1])
    dist2[:,:,2]=dist2[:,:,2]/np.sum(dist2[:,:,2])
    
    dist3=np.zeros((pieces,pieces,pieces,2));
    
    for q1 in range(pieces):
        for q2 in range(pieces):
            for q3 in range(pieces):
                k1=L1[q1];k2=L1[q2];k3=L1[q3]
                k4=L2[q1];k5=L2[q2];k6=L2[q3]
                count1=0;count2=0
                for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1) and d_x[i1,3]>=(k6-delta2) and d_x[i1,3]<=(k6+delta2):  
                        count1=count1+1
                   
                    if d_x[i1,2]>=(k4-delta2) and d_x[i1,2]<=(k4+delta2) and d_x[i1,3]>=(k5-delta2) and d_x[i1,3]<=(k5+delta2) and d_x[i1,1]>=(k3-delta1) and d_x[i1,1]<=(k3+delta1): 
                        count2=count2+1

                dist3[q1,q2,q3,0]=count1; dist3[q1,q2,q3,1]=count2;
            
    dist3[:,:,:,0]=dist3[:,:,:,0]/np.sum(dist3[:,:,:,0]); dist3[:,:,:,1]=dist3[:,:,:,1]/np.sum(dist3[:,:,:,1]);
    
    sum_f_1=0;sum_f_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            if dist2[k1,k2,1]!=0 and dist1[k2,1]!=0:
                sum_f_1=sum_f_1-dist2[k1,k2,1]*np.log2(dist2[k1,k2,1]/dist1[k2,1])

            if dist2[k1,k2,0]!=0 and dist1[k2,0]!=0:
                sum_f_2=sum_f_2-dist2[k1,k2,0]*np.log2(dist2[k1,k2,0]/dist1[k2,0])
    
    sum_s_1=0;sum_s_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            for k3 in range(pieces):
                if dist3[k1,k2,k3,1]!=0 and dist2[k3,k2,2]!=0:
                    sum_s_1=sum_s_1-dist3[k1,k2,k3,1]*np.log2(dist3[k1,k2,k3,1]/dist2[k3,k2,2])

                if dist3[k1,k2,k3,0]!=0 and dist2[k2,k3,2]!=0:
                    sum_s_2=sum_s_2-dist3[k1,k2,k3,0]*np.log2(dist3[k1,k2,k3,0]/dist2[k2,k3,2])
    
    en_1_2=sum_f_1-sum_s_1
    en_2_1=sum_f_2-sum_s_2
    
    return en_1_2, en_2_1

data_path = "data/exchange_rate.txt"
X = np.loadtxt(data_path, delimiter=',')
f1 = open('TE/ex.txt','a+')
A = np.eye(X.shape[1])
L = 0.8*X.shape[0]
L = int(L)
L1 = L

import time
import progressbar
from progress.bar import Bar
t = 0
bar = Bar('Processing', max=703, fill='@', suffix='%(percent)d%%')
for i in range(X.shape[1]):
    # if i < 100:
    #     continue
    for j in range(i+1,X.shape[1]):
        t += 1
        print('     ',t/7.03,'%\r')
        time.sleep(0.0000001)
        bar.next()
        # print('hello')
        te1,te2 = TE(X[:L,i],X[:L,j],50,L1-1)
        # f1 = open('TE/ex.txt', 'a+')
        if te1 >= te2:
            A[i,j] = te1-te2
            # f1.write(str(i) + '-' + str(j) + ':' + str(A[i, j]) + '\n')
            # f1.close()
        if te1 < te2:
            A[j,i] = te2-te1
            # f1.write(str(j) + '-' + str(i) + ':' + str(A[j, i]) + '\n')
            # f1.close()


bar.finish()
for i in range(X.shape[1]):
    for j in range(X.shape[1]):
        f1.write(str(A[i,j])+' ')
    f1.write('\n')
f1.close()
# A = np.loadtxt('TE/solar.txt')
# A = np.array(A, dtype=np.float32)