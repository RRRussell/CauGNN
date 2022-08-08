import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphAttentionLayer, SpGraphAttentionLayer
# from torch_geometric.nn import GCNConv
from layer import DeGINConv,DenseGCNConv,DenseSAGEConv,DenseGraphConv,rkGraphConv
from torch_geometric.nn import GATConv

class Model(nn.Module):

    def __init__(self, args,data):
        super(Model,self).__init__()
        self.use_cuda = args.cuda
        A = np.loadtxt(args.A)
        A = np.array(A,dtype=np.float32)
        # A = np.ones((args.n_e,args.n_e),np.int8)
        # A[A>0.05] = 1
        A = A/np.sum(A,0)
        A_new = np.zeros((args.batch_size,args.n_e,args.n_e),dtype=np.float32)
        for i in range(args.batch_size):
            A_new[i,:,:]=A

        self.A = torch.from_numpy(A_new).cuda()
        self.adjs = [self.A]
        self.num_adjs = args.num_adj
        if self.num_adjs>1:
            A = np.loadtxt(args.B)
            A = np.array(A, dtype=np.float32)
            # A = np.ones((args.n_e,args.n_e),np.int8)
            # A[A>0.05] = 1
            A = A / np.sum(A, 1)
            A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
            for i in range(args.batch_size):
                A_new[i, :, :] = A

            self.B = torch.from_numpy(A_new).cuda()
            A = np.ones((args.n_e,args.n_e),np.int8)
            A = A / np.sum(A, 1)
            A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
            for i in range(args.batch_size):
                A_new[i, :, :] = A
            self.C = torch.from_numpy(A_new).cuda()
            self.adjs = [self.A,self.B,self.C]

        # self.A = torch.from_numpy(A_new)
        self.n_e=args.n_e
        self.decoder = args.decoder
        self.attention_mode = args.attention_mode
        # if self.decoder != 'GAT':
        ##The hyper-parameters are applied to all datasets in all horizons
        self.conv1=nn.Conv2d(1, args.channel_size, kernel_size = (1,args.k_size[0]),stride=1)
        self.conv2=nn.Conv2d(1, args.channel_size, kernel_size = (1,args.k_size[1]),stride=1)
        self.conv3=nn.Conv2d(1, args.channel_size, kernel_size = (1,args.k_size[2]),stride=1)

        # self.maxpool1 = nn.MaxPool2d(kernel_size = (1,args.k_size[0]),stride=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1, args.k_size[1]), stride=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1, args.k_size[2]), stride=1)
        # self.dropout = nn.Dropout(p=0.1)
        d= (len(args.k_size)*(args.window) -sum(args.k_size)+ len(args.k_size))*args.channel_size
        skip_mode = args.skip_mode
        self.BATCH_SIZE=args.batch_size
        self.dropout = 0.1
        if self.decoder == 'GCN':
            self.gcn1 = DenseGCNConv(d, args.hid1)
            self.gcn2 = DenseGCNConv(args.hid1, args.hid2)
            self.gcn3 = DenseGCNConv(args.hid2, 1)

        if self.decoder == 'GNN':
        # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(d, args.hid1)
            self.gnn2 = DenseGraphConv(args.hid1, args.hid2)
            self.gnn3 = DenseGraphConv(args.hid2, 1)

        if self.decoder == 'rGNN':

            self.gc1 = rkGraphConv(self.num_adjs,d,args.hid1,self.attention_mode,aggr='mean')
            self.gc2 = rkGraphConv(self.num_adjs,args.hid1,args.hid2,self.attention_mode,aggr='mean')
            self.gc3 = rkGraphConv(self.num_adjs,args.hid2, 1, self.attention_mode, aggr='mean')

        self.hw = args.highway_window
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        if self.decoder == 'SAGE':
            self.sage1 = DenseSAGEConv(d,args.hid1)
            self.sage2 = DenseSAGEConv(args.hid1, args.hid2)
            self.sage3 = DenseSAGEConv(args.hid2, 1)

        if self.decoder == 'GIN':
            ginnn = nn.Sequential(
                nn.Linear(d,args.hid1),
                # nn.ReLU(True),
                # nn.Linear(args.hid1, args.hid2),
                nn.ReLU(True),
                nn.Linear(args.hid1,1),
                nn.ReLU(True)
            )
            self.gin = DeGINConv(ginnn)
        if self.decoder == 'GAT':
            self.gatconv1 = GATConv(d,args.hid1)
            self.gatconv2 = GATConv(args.hid1,args.hid2)
            self.gatconv3 = GATConv(args.hid2,1)

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode=="concat" else x2+x1
    def forward(self,x):
        c=x.permute(0,2,1)
        c=c.unsqueeze(1)
        # if self.decoder != 'GAT':
        a1=self.conv1(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        a2=self.conv2(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        a3=self.conv3(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        # a1 = self.dropout(a1)
        # a2 = self.dropout(a2)
        # a3 = self.dropout(a3)
        # x_conv = F.relu(torch.cat([a1,a2],2))
        x_conv = F.relu(torch.cat([a1, a2, a3], 2))
        # x_conv=F.relu(torch.cat([a1,a2,a3,a4,a5],2))
        # print(x_conv.shape)

        ##GCN1
        # x1=F.relu(torch.bmm(self.A,x_conv).bmm(self.w1))
        # x2=F.relu(torch.bmm(self.A,x1).bmm(self.w2))
        # x3=F.relu(torch.bmm(self.A,x2).bmm(self.w3))

        if self.decoder == 'GCN':
            # x1 = F.relu(self.gcn1(x_conv,self.A))
            x1 = F.relu(self.gcn1(x_conv, self.A))
            x2 = F.relu(self.gcn2(x1,self.A))
            x3 = self.gcn3(x2,self.A)
            x3 = x3.squeeze()

        if self.decoder == 'GNN':
        # x0 = F.relu(self.gnn0(x_conv,self.A))
            x1 = F.relu(self.gnn1(x_conv,self.A))
            x2 = F.relu(self.gnn2(x1,self.A))
            x3 = self.gnn3(x2,self.A)
            x3 = x3.squeeze()
        if self.decoder == 'rGNN':
            x1 = F.relu(self.gc1(x_conv,self.adjs))
            # x1 = F.dropout(x1, self.dropout)
            x2 = F.relu(self.gc2(x1, self.adjs))
            x3 = F.relu(self.gc3(x2, self.adjs))
            # x3 = F.dropout(x2, self.dropout)
            x3 = x3.squeeze()

        if self.decoder == 'SAGE':
            x1 = F.relu(self.sage1(x_conv,self.A))
            x2 = F.relu(self.sage2(x1,self.A))
            x3 = F.relu(self.sage3(x2,self.A))
            x3 = x3.squeeze()

        if self.decoder == 'GIN':
            x3 = F.relu(self.gin(x_conv, self.A))
            x3 = x3.squeeze()

        if self.decoder == 'GAT':
            x1 = F.relu(self.gatconv1(x_conv,self.edge_index))
            x2 = F.relu(self.gatconv2(x1,self.edge_index))
            x3 = F.relu(self.gatconv3(x2,self.edge_index))
            x3 = x3.squeeze()

        if self.hw>0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1)
            z = self.highway(z)
            z = z.squeeze(2)
            x3 = x3 + z
        return x3
