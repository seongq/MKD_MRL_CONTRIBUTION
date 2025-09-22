import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
# import ipdb
# from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops
from graphgcn import GraphGCN


class GCN(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, num_L=3, num_K=4,
                args=None):
        super(GCN, self).__init__()
        self.args=args
        self.MRL = self.args.MRL
        self.MRL_efficient = self.args.MRL_efficient
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.n_dim = n_dim
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.dropout_ = nn.Dropout(self.dropout)
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker

        self.fc_l = nn.Linear(n_dim, nhidden)
        self.fc_a = nn.Linear(n_dim, nhidden)
        self.fc_v = nn.Linear(n_dim, nhidden)
        self.fc = nn.Linear(nhidden, nhidden)
        self.num_K =  num_K
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), GraphGCN(nhidden, nhidden,args=self.args))

    def forward(self, a, v, l, dia_len, qmask, epoch, two_modals=None):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            l += spk_emb_vector
            v += spk_emb_vector
            a += spk_emb_vector
      
        l = self.fc_l(l)
        l = self.dropout_(l)
        l = self.act_fn(l)
        
        v = self.fc_v(v)
        v = self.dropout_(v)
        v - self.act_fn(v)
        
        a = self.fc_a(a)
        a = self.dropout_(a)
        a = self.act_fn(a)
        
        
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        # print(gnn_features.size(), "gnn_feature_size()")
    
        x1 = self.fc(gnn_features)  
        x1 = self.dropout_(x1)
        x1 = self.act_fn(x1)
        out = x1
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([gnn_features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        
        return out1
        
            

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals, two_modals=None):
        num_modality = len(modals)
        node_count = 0
        index =[]
        tmp = []
        # print(two_modals)
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes] 
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])     
                    
                
            
            # print(len(Gnodes))
                
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
            lengths = 0
            
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).to("cuda:0")
        return edge_index, features
