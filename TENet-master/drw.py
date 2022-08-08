# # 载入模块
import numpy as np
import matplotlib.pyplot as plt
from result_matrix import *

# -*- coding: utf-8 -*-

# name_list = ['0', '1', '', 'Sunday']
# num_list = get_rmse()
# num_list1 = get_rse()
# x = list(range(len(num_list)))
# total_width, n = 3, 10
# width = total_width / n
# plt.ylim((0.009, 0.024))
# plt.bar(x, num_list, width=width, label='boy', fc='y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# # plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
# plt.bar(x, num_list1, width=width, label='girl', fc='r')
# plt.legend()
# plt.show()

##-----读取数据----------
# data = open("E:/Draw/data.csv").readlines()
name = []
VI = []
VC = []
# for i in range(len(data)):
#     data[i] = data[i].strip('\n').split(',')
#     name.append(data[i][0])
#     VI.append(eval(data[i][1]))
#     VC.append(eval(data[i][2]))
name = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
VI = get_rae()
VC = get_corr()
VII = get_mae()
##-----绘制右边柱子----------
##w是柱子的宽度
w = 1
##生成绘图对象
fig = plt.figure()
fig.set_size_inches(10,2.5)
##将画布划分成1行1列，在第一个方格中绘制
ax1 = fig.add_subplot(111)
##隐藏上边框
ax1.spines['top'].set_color('none')
##设置刻度范围
ax1.set_ylim(0.92, 1)
ax1.set_ylabel('CORR')
##设置x轴为下边框
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xlabel('GCN hidden size')
##将x轴移动至0刻度处
ax1.spines['bottom'].set_position(('data',0.5*max(VC) ))
##设置x轴的标签，3个单位长度间隔，和柱子间距一致
# ax1.set_xticks(np.arange(0,len(name)*4,4))
# ax1.set_xticklabels(name, ha='left', fontsize=9)
##设置柱子间距
idx = np.arange(w, len(name) * 4 + w, 4)
##生成柱状图，VI是数据列，width是柱子宽度
p1 = plt.bar(idx, VC, width=w,label = 'corr', color='#48D1CC')

##-----绘制左边柱子----------
##共享x坐标，这是关键
ax2 = ax1.twinx()
ax2.spines['bottom'].set_position(('data',0.5*max(VC)))
ax2.set_ylim(0,0.025)
ax2.set_ylabel('RAE & MAE')

# ax2.set_xticks(np.arange(0, len(name) * 4, 4))
# ax2.set_xticklabels(name, ha='left', fontsize=9)

p2 = plt.bar(idx - w, VI,label = 'rae', color='#CD853F',width=w)

ax3 = ax1.twinx()
# ax3.spines['bottom'].set_position(('data',0.5*max(VC)))
ax3.set_ylim(0.1*max(VI), 1*max(VI))
ax3.set_yticks([])
ax3.set_xticks(np.arange(0, len(name) * 4, 4))
ax3.set_xticklabels(name, ha='left', fontsize=9)
p3 = plt.bar(idx+w,VII,width=w,label = 'mae', color='#008000')

plt.legend([p1, p2,p3], ['corr', 'rae','mae'],loc='center', bbox_to_anchor=(-0.105, 0.9),fontsize = '9')
plt.savefig('1.png', dpi=600)
plt.show()
#Parameter sensitivity test results. TEGNN shows similar performance under different settings of hidden sizes in GNN layer when forecasting   $\{t+5\}$value on Exchange_rate dataset.




