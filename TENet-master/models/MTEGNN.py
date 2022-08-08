import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphAttentionLayer, SpGraphAttentionLayer
# from torch_geometric.nn import GCNConv
from layer import DeGINConv,DenseGCNConv,DenseSAGEConv,DenseGraphConv,rkGraphConv
from torch_geometric.nn import GATConv
from MTlayer import *
import pickle
import os
import scipy.sparse as sp
from scipy.sparse import linalg
from torch.autograd import Variable

class Model(nn.Module):

    def __init__(self, args,gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(Model,self).__init__()
        self.use_cuda = args.cuda
        A = np.loadtxt(args.A)
        A = np.array(A,dtype=np.float32)
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

        if self.decoder == 'rGNN':

            self.gc1 = rkGraphConv(self.num_adjs,d,args.hid1,self.attention_mode,aggr='mean')
            self.gc2 = rkGraphConv(self.num_adjs,args.hid1,args.hid2,self.attention_mode,aggr='mean')
            self.gc3 = rkGraphConv(self.num_adjs,args.hid2, 1, self.attention_mode, aggr='mean')

            self.attention = torch.nn.Parameter(nn.init.uniform_(torch.FloatTensor(2)))
            self.gcn_true = gcn_true
            self.buildA_true = buildA_true
            self.num_nodes = num_nodes
            self.dropout = dropout
            self.predefined_A = predefined_A
            self.filter_convs = nn.ModuleList()
            self.gate_convs = nn.ModuleList()
            self.residual_convs = nn.ModuleList()
            self.skip_convs = nn.ModuleList()
            self.gconv1 = nn.ModuleList()
            self.gconv2 = nn.ModuleList()
            self.norm = nn.ModuleList()
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                        static_feat=static_feat)

            self.seq_length = seq_length
            kernel_size = 7
            if dilation_exponential > 1:
                self.receptive_field = int(
                    1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                self.receptive_field = layers * (kernel_size - 1) + 1

            for i in range(1):
                if dilation_exponential > 1:
                    rf_size_i = int(
                        1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
                else:
                    rf_size_i = i * layers * (kernel_size - 1) + 1
                new_dilation = 1
                for j in range(1, layers + 1):
                    if dilation_exponential > 1:
                        rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (
                                    dilation_exponential - 1))
                    else:
                        rf_size_j = rf_size_i + j * (kernel_size - 1)

                    self.filter_convs.append(
                        dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                    self.gate_convs.append(
                        dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                    self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                         out_channels=residual_channels,
                                                         kernel_size=(1, 1)))
                    if self.seq_length > self.receptive_field:
                        self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                         out_channels=skip_channels,
                                                         kernel_size=(1, self.seq_length - rf_size_j + 1)))
                    else:
                        self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                         out_channels=skip_channels,
                                                         kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                    if self.gcn_true:
                        self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                        self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                    if self.seq_length > self.receptive_field:
                        self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                                   elementwise_affine=layer_norm_affline))
                    else:
                        self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                                   elementwise_affine=layer_norm_affline))

                    new_dilation *= dilation_exponential

            self.layers = layers
            self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)
            self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                        out_channels=out_dim,
                                        kernel_size=(1, 1),
                                        bias=True)
            if self.seq_length > self.receptive_field:
                self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                       bias=True)
                self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                       kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

            else:
                self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                       kernel_size=(1, self.receptive_field), bias=True)
                self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                       bias=True)

            self.idx = torch.arange(self.num_nodes).to(device)

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode=="concat" else x2+x1
    def forward(self,input,idx=None):
        c=input.permute(0,2,1)
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

        if self.decoder == 'rGNN':
            x1 = F.relu(self.gc1(x_conv,self.adjs))
            # x1 = F.dropout(x1, self.dropout)
            x2 = F.relu(self.gc2(x1, self.adjs))
            x3 = F.relu(self.gc3(x2, self.adjs))
            # x3 = F.dropout(x2, self.dropout)
            x3 = x3.squeeze()

            input = input.unsqueeze(1).permute(0, 1, 3, 2)
            seq_len = input.size(3)
            assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

            if self.seq_length < self.receptive_field:
                input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

            if self.gcn_true:
                if self.buildA_true:
                    if idx is None:
                        adp = self.gc(self.idx)
                    else:
                        adp = self.gc(idx)
                else:
                    adp = self.predefined_A

            x = self.start_conv(input)
            skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
            for i in range(self.layers):
                residual = x
                filter = self.filter_convs[i](x)
                filter = torch.tanh(filter)
                gate = self.gate_convs[i](x)
                gate = torch.sigmoid(gate)
                x = filter * gate
                x = F.dropout(x, self.dropout, training=self.training)
                s = x
                s = self.skip_convs[i](s)
                skip = s + skip
                if self.gcn_true:
                    x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
                else:
                    x = self.residual_convs[i](x)

                x = x + residual[:, :, :, -x.size(3):]
                if idx is None:
                    x = self.norm[i](x, self.idx)
                else:
                    x = self.norm[i](x, idx)

            skip = self.skipE(x) + skip
            x = F.relu(skip)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x).squeeze()
            attention = F.softmax(self.attention)
            x3 = attention[0]*x3 + attention[1]*x

        if self.hw>0:
            z = input[:, -self.hw:, :]
            z = z.permute(0, 2, 1)
            z = self.highway(z)
            z = z.squeeze(2)
            x3 = x3 + z
        return x3
