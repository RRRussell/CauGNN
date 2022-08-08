import matplotlib.pyplot as plt
import sys,pprint
from gexf import Gexf
import networkx as nx
import random
import numpy as np
import os
path = "/home/jiangnanyida/dzh/GraphMatching_BA/dzh_mini_data/IMDBMulti/all/"
l = []
with open('graph.txt','w') as f1:
    for f in os.listdir(path):
        graph = nx.read_gexf(path+f)
        l.append((int(len(graph.nodes)),int(len(graph.edges))))
    l.sort(key=lambda tup: tup[0])
    for i in l:
        f1.write(str(i)+'\n')
    f1.close()



