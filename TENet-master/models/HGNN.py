import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphAttentionLayer, SpGraphAttentionLayer
# from torch_geometric.nn import GCNConv
from layer import DeGINConv,DenseGCNConv,DenseSAGEConv,DenseGraphConv,rkGraphConv


class Model(nn.Module):

    def __init__(self, args,data):
        super(Model,self).__init__()
        self.use_cuda = args.cuda
        # A = np.loadtxt(args.A)
        # A = np.array(A,dtype=np.float32)
        A = np.ones((data.num_nodes,data.num_nodes), np.int8)
        # A[A>0.05] = 1
        A = A / np.sum(A, 1)
        A_new = np.zeros((1,data.num_nodes,data.num_nodes), dtype=np.float32)
        A_new[1, :, :] = A
        self.A = torch.from_numpy(A_new).cuda()

        self.num_adjs = args.num_adj

        self.n_e=args.n_e
        self.decoder = args.decoder
        self.attention_mode = args.attention_mode

        self.conv1=nn.Conv2d(1, 12, kernel_size = (1,3),stride=1)
        self.conv2=nn.Conv2d(1, 12, kernel_size = (1,5),stride=1)
        self.conv3=nn.Conv2d(1, 12, kernel_size = (1,7),stride=1)

        d= (len(args.k_size)*(args.window) -sum(args.k_size)+ len(args.k_size))*args.channel_size

        skip_mode = args.skip_mode
        self.BATCH_SIZE=data.batch_size
        self.dropout = 0.1
        if self.decoder == 'GCN':
            self.gcn1 = DenseGCNConv(d, args.hid1)
            self.gcn2 = DenseGCNConv(args.hid1, args.hid2)
            self.gcn3 = DenseGCNConv(args.hid2, 1)

        if self.decoder == 'GNN':
        # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(d, args.hid1)
            self.gnn2 = DenseGraphConv(args.hid1, args.hid2)
            self.gnn3 = DenseGraphConv(args.hid2, data.num_class)


    def forward(self,x):

        c=x.permute(0,2,1)
        c=c.unsqueeze(1)
        a1=self.conv1(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        a2=self.conv2(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        a3=self.conv3(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)

        x_conv = F.relu(torch.cat([a1, a2, a3], 2))

        if self.decoder == 'GCN':
            x1 = F.relu(self.gcn1(x_conv,self.A))
            x2 = F.relu(self.gcn2(x1,self.A))
            x3 = self.gcn3(x2,self.A)
            x3 = x3.squeeze()

        if self.decoder == 'GNN':
        # x0 = F.relu(self.gnn0(x_conv,self.A))
            x1 = F.relu(self.gnn1(x_conv,self.A))
            x2 = F.relu(self.gnn2(x1,self.A))
            x3 = self.gnn3(x2,self.A)
            # x3 = x3.squeeze()
            x3 = F.log_softmax(x3, dim=1)

        return x3

    def get_loss(self, output, labels, idx_lst):
        return F.nll_loss(output[idx_lst], labels[idx_lst])
