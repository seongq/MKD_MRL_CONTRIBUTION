

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from itertools import permutations

def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class GraphGCN(MessagePassing):
    def __init__(self, in_channels ):
        super(GraphGCN, self).__init__(aggr='add') 

        self.gate = torch.nn.Linear(2*in_channels, 1)
    def forward(self, x, edge_index):
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x )

    def message(self, x_i, x_j, edge_index, size):
    
        row, col = edge_index
        deg = degree(col, size[0], dtype=x_j.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))#e.g.[135090, 1]

        return norm.view(-1, 1) * (x_j) *alpha_g

    def update(self, aggr_out):
      
        return aggr_out


class UNI_GCN(nn.Module):
    def __init__(self, n_dim, nhidden=512, num_K=4):
        super(UNI_GCN, self).__init__()
        

        
        
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)         
        self.num_K =  num_K




        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), GraphGCN(nhidden))

    def forward(self, emotions_feat, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)

        #---------------------------------------
        gnn_edge_index, gnn_features = self.create_gnn_index(emotions_feat, dia_len)
        x1 = self.fc1(gnn_features)  
        out = x1
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
     

        out1 = out2
        #---------------------------------------
        return out1


    def create_gnn_index(self, emotions_feat, dia_len, self_loops=False):
        assert sum(dia_len) == emotions_feat.size(0), "dia_len 합 != 노드 수"
        device = emotions_feat.device
        pieces = []
        off = 0

        for L in dia_len:
            if L <= 0:
                continue
            idx = torch.arange(off, off+L, device=device)
            src = idx.repeat_interleave(L)   # [L*L]
            dst = idx.repeat(L)              # [L*L]
            mask = src != dst                # self-loop 제거
            e = torch.stack([src[mask], dst[mask]], dim=0)  # [2, L*(L-1)]
            if self_loops:
                loop = torch.stack([idx, idx], dim=0)
                e = torch.cat([e, loop], dim=1)
            pieces.append(e)
            off += L

        edge_index = torch.cat(pieces, dim=1) if pieces else torch.empty(2, 0, dtype=torch.long, device=device)
        
        return edge_index, emotions_feat



class Loss_Function(nn.Module):
    def __init__(self, class_weight=False, weight=None , loss_type = "Focal", dataset="MELD", focal_prob="softmax", gamma = 2.5, alpha = 1, reduction = 'mean'):
        super(Loss_Function, self).__init__()
        self.loss_type = loss_type
        self.class_weight = class_weight
        self.weight = weight
        self.dataset = dataset
        if self.loss_type == "Focal":
            self.focal_prob = focal_prob
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction
            self.elipson = 0.000001
            self.loss_function = FocalLoss(self.gamma, self.alpha, self.reduction, self.focal_prob)

        elif self.loss_type == "NLL":
            if self.class_weight:
                if self.dataset=="IEMOCAP":
                    self.loss_function = nn.NLLLoss(weight)
                elif self.dataset=="MELD":
                    self.loss_function = nn.NLLLoss()
            else:
                self.loss_function = nn.NLLLoss()

                
    def forward(self, logits, labels):
        
        if self.loss_type == "Focal":
            
            return self.loss_function(logits, labels)
        elif self.loss_type == "NLL":
            return self.loss_function(F.log_softmax(logits,dim=-1), labels)

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, reduction = 'mean', focal_prob='prob',args = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.elipson = 0.000001
        self.args = args
        self.focal_prob = focal_prob
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        device = logits.device
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length], device=device).scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        if self.focal_prob == 'prob':
            prob = torch.exp(log_p)    
        elif self.focal_prob == 'log_prob':
            prob = log_p
        else:
            raise ValueError("Unknown focal_prob type")
        
        pt = label_onehot * prob
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        else:  # "none"
            return fl

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss



def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)  

    if not no_cuda:
        node_features = node_features.to("cuda:0")
    return node_features


class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim, num_heads):
        super(MultiHeadCrossModalAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 多头投影层，将图像和文本分别映射到多头的 Q、K、V
        self.img_query_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_key_proj = nn.Linear(txt_dim, hidden_dim)
        self.txt_value_proj = nn.Linear(txt_dim, hidden_dim)

        # 最后的输出线性变换
        self.output_proj = nn.Linear(hidden_dim, img_dim)

    def forward(self, img_features, txt_features):
        """
        :param img_features: [batch_size, num_regions, img_dim]
        :param txt_features: [batch_size, num_words, txt_dim]
        :return: 融合后的特征
        """
        B, R, _ = img_features.shape  # B: batch_size, R: num_regions
        _, W, _ = txt_features.shape  # W: num_words

        # 线性投影得到 Q、K、V，并 reshape 为多头格式
        Q = self.img_query_proj(img_features).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)  
        K = self.txt_key_proj(txt_features).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  
        V = self.txt_value_proj(txt_features).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  

        # 计算注意力权重: Q·K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        attention_weights = F.softmax(attention_scores, dim=-1)  

        # 加权求和得到上下文表示
        attended_features = torch.matmul(attention_weights, V)  

        # 合并多头的结果
        attended_features = attended_features.transpose(1, 2).contiguous().view(B, R, -1)  

        # 输出线性变换
        output = img_features + self.output_proj(attended_features)  
        return output
        

class Model(nn.Module):

    def __init__(self, 
                 MKD = False,
                 use_speaker_embedding=True,
                 n_speakers=2,
                 n_classes=7, 
                 dropout=0.5,
                 no_cuda=False, 
                 D_m_v=512,
                 D_m_a=100,
                 D_m_l=1024,
                 hidden_dim = 1024,
                 dataset='IEMOCAP',
                 MRL=False,
                 MRL_efficient=False,
                 mrl_num_partition=1,
                 calib=False,
                 MKD_last_layer =False,
                using_MHA = False,
                number_of_heads = 2,
                using_graph = False,
                using_multimodal_graph = False,
                num_K = 4,
                graph_hidden_dim = 512,
                  args=None):
        
        super(Model, self).__init__()
        self.graph_hideen_dim = graph_hidden_dim
        self.using_graph = using_graph
        self.using_multimodal_graph = using_multimodal_graph
        self.num_K = num_K
        
        self.using_MHA = using_MHA
        self.number_of_heads = number_of_heads

        self.MKD_last_layer = MKD_last_layer
        self.calib = calib
        self.MRL = MRL
        self.MRL_efficient = MRL_efficient
        self.mrl_num_partition = mrl_num_partition

        
        self.MKD = MKD
        self.n_classes = n_classes
        self.args = args
        self.no_cuda = no_cuda
        self.n_speakers = n_speakers
        self.dropout = dropout

        self.hidden_dim = hidden_dim        
        self.dataset = dataset
        self.stablizing = args.stablizing
        
        # BN for four textual streams
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)
        self.use_speaker_embedding = use_speaker_embedding 
        if self.use_speaker_embedding:
            self.speaker_embeddings = nn.Embedding(n_speakers,  self.hidden_dim)
        # Encoders per modality
        hidden_a = self.hidden_dim
        self.linear_a = nn.Linear(D_m_a, hidden_a)
        self.enc_a = nn.LSTM(input_size=hidden_a, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)
           

        hidden_v = self.hidden_dim
        self.linear_v = nn.Linear(D_m_v, hidden_v)
        self.enc_v = nn.LSTM(input_size=hidden_v, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)
           

        hidden_l = self.hidden_dim
        self.linear_l = nn.Linear(D_m_l, hidden_l)
        self.enc_l = nn.LSTM(input_size=hidden_l, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)
           
        if self.using_MHA:
            self.attention_a = MultiHeadCrossModalAttention(self.hidden_dim,self.hidden_dim,self.hidden_dim, num_heads = self.number_of_heads)
            self.attention_v = MultiHeadCrossModalAttention(self.hidden_dim,self.hidden_dim,self.hidden_dim, num_heads = self.number_of_heads)
            self.attention_l = MultiHeadCrossModalAttention(self.hidden_dim,self.hidden_dim,self.hidden_dim, num_heads = self.number_of_heads)
            
            

        
        # 결합 후 안정화용
        self.ln_a = nn.LayerNorm(self.hidden_dim)
        self.ln_v = nn.LayerNorm(self.hidden_dim)
        self.ln_l = nn.LayerNorm(self.hidden_dim)
        self.dropout_ = nn.Dropout(self.dropout)


        self.num_modals = 3
        
        
        
        if self.using_graph:
            if not self.using_multimodal_graph:
                self.gcn_a = UNI_GCN(self.hidden_dim, self.graph_hideen_dim, self.num_K)
                self.gcn_v = UNI_GCN(self.hidden_dim, self.graph_hideen_dim, self.num_K)
                self.gcn_l = UNI_GCN(self.hidden_dim, self.graph_hideen_dim, self.num_K)  
        
        

        self.smax_fc = nn.Linear((self.hidden_dim)*self.num_modals, self.n_classes, bias=False)
        self.last_feature_dimension = self.smax_fc.in_features
        if self.MRL:
            temp = (self.last_feature_dimension // 3)//2
            self.mrl_sizeset = []
            if not self.MRL_efficient:
                self.mrl_layers = nn.ModuleList([])
            while temp > 0 and isinstance(temp, int):
                self.mrl_sizeset.append(temp)
                if not self.MRL_efficient:
                    self.mrl_layers.append(nn.Linear(temp*3, self.n_classes, bias=False))
                temp = temp //2
                    
            
            self.mrl_sizeset = self.mrl_sizeset[:self.mrl_num_partition]
            # print(self.mrl_sizeset)
            # print(self.mrl_num_partition)

        print(self.hidden_dim)
        if self.MKD:
            self.student_a = nn.Linear((self.hidden_dim), self.n_classes, bias=False)
            self.student_v = nn.Linear((self.hidden_dim), self.n_classes, bias=False)
            self.student_l = nn.Linear((self.hidden_dim), self.n_classes, bias=False)
            self.uni_a = Unimodal_MODEL(
                use_speaker_embedding=use_speaker_embedding,
                n_speakers=n_speakers,
                n_classes=self.n_classes,
                dropout=dropout,
                no_cuda=no_cuda,
                D_m_a=D_m_a,
                hidden_dim=hidden_dim,
                dataset=dataset,
                uni_modality="a",
                using_MHA = self.using_MHA ,
                number_of_heads = self.number_of_heads,
                using_graph = self.using_graph,
                    
                    num_K = self.num_K,
                    graph_hidden_dim = self.graph_hideen_dim,
                args=args
                )
            self.uni_v = Unimodal_MODEL(
                use_speaker_embedding=use_speaker_embedding, 
                n_speakers=n_speakers,
                n_classes=self.n_classes,
                dropout=dropout,
                no_cuda=no_cuda,
                D_m_v=D_m_v,
                hidden_dim=hidden_dim,
                dataset=dataset,
                uni_modality="v",
                using_MHA = self.using_MHA ,
                number_of_heads = self.number_of_heads,
                    using_graph = self.using_graph,
                    
                    num_K = self.num_K,
                    graph_hidden_dim = self.graph_hideen_dim,
                args=args
            )
                
            self.uni_l = Unimodal_MODEL(
                use_speaker_embedding=use_speaker_embedding,
                n_speakers=n_speakers,
                n_classes=self.n_classes,
                dropout=dropout,
                no_cuda=no_cuda,
                D_m_l=D_m_l,
                hidden_dim=hidden_dim,
                dataset=dataset,
                uni_modality="l",
                using_MHA = self.using_MHA ,
                number_of_heads = self.number_of_heads,
                using_graph = self.using_graph,
                    
                    num_K = self.num_K,
                    graph_hidden_dim = self.graph_hideen_dim,
                args=args
                )

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        logits_modal = {}
        logits_uni_modal_student ={}
        logits_MKD_teacher = {}
        logits_MRL_feature = {}
        logits_uni_modal = {}
        if self.MKD:
            logits_MKD_teacher = self.MKD_teacher_forward(U, qmask, umask, seq_lengths, U_a, U_v, epoch)
            
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        if r1.size(0)==1:
            pass
        else:
            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U_l = (r1 + r2 + r3 + r4)/4

        U_a = self.linear_a(U_a)
        U_v = self.linear_v(U_v)
        U_l = self.linear_l(U_l)
        
        if self.args.use_speaker_embedding:
            
            spk_idx = torch.argmax(qmask, dim=-1)
            spk_emb_vector = self.speaker_embeddings(spk_idx)
            U_l = U_l + spk_emb_vector
            U_v = U_v + spk_emb_vector
            U_a = U_a + spk_emb_vector
        U_a = self.dropout_(U_a)
        U_a = nn.ReLU()(U_a)
        

        # Visual
        U_v = self.dropout_(U_v)
        U_v = nn.ReLU()(U_v)
        

        # Language
        U_l = self.dropout_(U_l)
        U_l = nn.ReLU()(U_l)
        
        emotions_a, _ = self.enc_a(U_a)
        emotions_v, _ = self.enc_v(U_v)
        emotions_l, _ = self.enc_l(U_l)
        # print(emotions_a.size())
        if self.using_MHA:
            emotions_a = self.attention_a(emotions_a, emotions_a)
            emotions_v = self.attention_v(emotions_v, emotions_v)
            emotions_l = self.attention_l(emotions_l, emotions_l)
        # 결합 후 안정화 (모든 모드에 동일 적용 권장)

        # print(emotions_a.size())
        if self.stablizing:
            emotions_a = self.ln_a(self.dropout_(emotions_a))
            emotions_v = self.ln_v(self.dropout_(emotions_v))
            emotions_l = self.ln_l(self.dropout_(emotions_l))
        else:
            emotions_a = self.dropout_(emotions_a)
            emotions_v = self.dropout_(emotions_v)
            emotions_l = self.dropout_(emotions_l)
     
        emotions_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
        emotions_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
        emotions_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)

        if self.using_graph:
            if not self.using_multimodal_graph:
                emotions_a = self.gcn_a(emotions_a, seq_lengths, qmask, epoch)
                emotions_v = self.gcn_v(emotions_v, seq_lengths, qmask, epoch)
                emotions_l =  self.gcn_l(emotions_l, seq_lengths, qmask, epoch)  
        
        
        if self.calib:
            logits_modal = self.modal_comb_forward(emotions_a, emotions_v, emotions_l)
        
        if self.MKD:
            logits_uni_modal_student['a']=self.student_a(emotions_a)
            logits_uni_modal_student['v']=self.student_v(emotions_v)
            logits_uni_modal_student['l']=self.student_l(emotions_l)
            if self.MKD_last_layer:
                if self.calib:
                    logits_uni_modal['a'] = logits_modal['a']
                    logits_uni_modal['v'] = logits_modal['v']
                    logits_uni_modal['l'] = logits_modal['l']
                else:
                    logits_uni_modal = self.modal_uni_forward(emotions_a, emotions_v, emotions_l)
                    
        emotions_feature = torch.cat([emotions_a, emotions_v, emotions_l], dim=-1)
        emotions_feature = nn.ReLU()(emotions_feature)
        emotions_feature = self.dropout_(emotions_feature)
        if self.MRL:
            for i, size_ in enumerate(self.mrl_sizeset):
                temp_features = torch.cat([emotions_feature[:,0:size_],
                                              emotions_feature[:, self.last_feature_dimension//3*1:self.last_feature_dimension//3*1+size_], 
                                              emotions_feature[:,self.last_feature_dimension//3*2:self.last_feature_dimension//3*2+size_]], 
                                             dim=-1)
                # print(temp_features.size())

                
                if not self.MRL_efficient:
                    logits_MRL_feature[size_] = self.mrl_layers[i](temp_features)
                else:
                    W_temp = torch.cat([self.smax_fc.weight[:,0:size_], 
                                        self.smax_fc.weight[:, self.last_feature_dimension//3*1:self.last_feature_dimension//3*1+size_], 
                                        self.smax_fc.weight[:,self.last_feature_dimension//3*2:self.last_feature_dimension//3*2+size_]], dim=-1) 
                    logits_MRL_feature[size_] = F.linear(temp_features, W_temp)
                

        
        logits = self.smax_fc(emotions_feature)
        return logits, logits_uni_modal_student, logits_MKD_teacher, logits_MRL_feature, logits_modal, logits_uni_modal
    
    def MKD_teacher_forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):
        logits_MKD = {}
        logits_MKD['a'] = self.uni_a(U=None, qmask=qmask, umask=umask, seq_lengths=seq_lengths, U_a=U_a, U_v=None, epoch=epoch)
        logits_MKD['v'] = self.uni_v(U=None, qmask=qmask, umask=umask, seq_lengths=seq_lengths, U_a=None, U_v=U_v, epoch=epoch)
        logits_MKD['l'] = self.uni_l(U=U, qmask=qmask, umask=umask, seq_lengths=seq_lengths, U_a=None, U_v=None, epoch=epoch)
        return logits_MKD
    
    def modal_comb_forward(self, emotions_a, emotions_v,emotions_l):
        logits_modal = {}
 
        logits_modal['a'] = F.linear(emotions_a, self.smax_fc.weight[:,0:self.hidden_dim].contiguous())
        logits_modal['v'] = F.linear(emotions_v, self.smax_fc.weight[:,self.hidden_dim:2*self.hidden_dim].contiguous())
        logits_modal['l'] = F.linear(emotions_l, self.smax_fc.weight[:,2*self.hidden_dim:].contiguous())

        emotions_av = torch.cat([emotions_a, emotions_v], dim=-1)
        emotions_al = torch.cat([emotions_a, emotions_l], dim=-1)
        emotions_vl = torch.cat([emotions_v, emotions_l], dim=-1)
        w_av = self.smax_fc.weight[:,0:self.hidden_dim*2].contiguous()
        w_al = torch.cat([self.smax_fc.weight[:,0:self.hidden_dim],self.smax_fc.weight[:,2*self.hidden_dim:]], dim=-1).contiguous()
        w_vl = self.smax_fc.weight[:, self.hidden_dim:].contiguous()

        logits_modal['av'] = F.linear(emotions_av, w_av)
        logits_modal['al'] = F.linear(emotions_al, w_al)
        logits_modal['vl'] = F.linear(emotions_vl, w_vl)
        return logits_modal
    def modal_uni_forward(self, emotions_a, emotions_v,emotions_l):
        logits_modal = {}
 
        logits_modal['a'] = F.linear(emotions_a, self.smax_fc.weight[:,0:self.hidden_dim].contiguous())
        logits_modal['v'] = F.linear(emotions_v, self.smax_fc.weight[:,self.hidden_dim:2*self.hidden_dim].contiguous())
        logits_modal['l'] = F.linear(emotions_l, self.smax_fc.weight[:,2*self.hidden_dim:].contiguous())

        
        return logits_modal
        

class Unimodal_MODEL(nn.Module):

    def __init__(self, 
                 use_speaker_embedding=True,
                 n_speakers=2,
                 n_classes=7, 
                 dropout=0.5,
                 no_cuda=False, 
                 D_m_v=512,
                 D_m_a=100,
                 D_m_l=1024,
                 hidden_dim = 1024,
                 dataset='IEMOCAP',
                 uni_modality = "l",
                 using_MHA = False,
                number_of_heads = 2,
                using_graph = False,
                num_K = 4,
                graph_hidden_dim = 512,
                 args=None):
        
        super(Unimodal_MODEL, self).__init__()
        
        self.using_graph = using_graph
        self.graph_hideen_dim = graph_hidden_dim
        self.num_K = num_K
        
        
        self.n_classes = n_classes
        self.args = args
        self.no_cuda = no_cuda
        self.n_speakers = n_speakers
        self.dropout = dropout

        self.hidden_dim = hidden_dim        
        self.dataset = dataset
        self.stablizing = args.stablizing
        
        self.number_of_heads = number_of_heads
        self.using_MHA = using_MHA
        
        self.use_speaker_embedding = use_speaker_embedding 
        if self.use_speaker_embedding:
            self.speaker_embeddings = nn.Embedding(n_speakers,  self.hidden_dim)
        
        
        self.uni_modality = uni_modality
        
        
        if self.uni_modality == "l":
            # BN for four textual streams
            self.normBNa = nn.BatchNorm1d(1024, affine=True)
            self.normBNb = nn.BatchNorm1d(1024, affine=True)
            self.normBNc = nn.BatchNorm1d(1024, affine=True)
            self.normBNd = nn.BatchNorm1d(1024, affine=True)
            
            
            hidden_l = self.hidden_dim
            self.linear_l = nn.Linear(D_m_l, hidden_l)
            self.enc_l = nn.LSTM(input_size=hidden_l, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)

            if self.using_MHA:
                self.attention_l = MultiHeadCrossModalAttention(hidden_l,hidden_l,hidden_l, num_heads = self.number_of_heads)
            self.ln_l = nn.LayerNorm(self.hidden_dim)

        
        elif self.uni_modality == "a":
            hidden_a = self.hidden_dim
            self.linear_a = nn.Linear(D_m_a, hidden_a)
            self.enc_a = nn.LSTM(input_size=hidden_a, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)
            if self.using_MHA:
                self.attention_a = MultiHeadCrossModalAttention(hidden_a,hidden_a,hidden_a, num_heads = self.number_of_heads)
            
            self.ln_a = nn.LayerNorm(self.hidden_dim)

        elif self.uni_modality == "v":
            hidden_v = self.hidden_dim
            self.linear_v = nn.Linear(D_m_v, hidden_v)
            self.enc_v = nn.LSTM(input_size=hidden_v, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)

            if self.using_MHA:
                self.attention_v = MultiHeadCrossModalAttention(hidden_v,hidden_v,hidden_v, num_heads = self.number_of_heads)
            self.ln_v = nn.LayerNorm(self.hidden_dim)


        if self.using_graph:        
            self.gcn = UNI_GCN(self.hidden_dim, self.graph_hideen_dim, self.num_K)       
        self.dropout_ = nn.Dropout(self.dropout)


        self.num_modals = 3

        self.smax_fc = nn.Linear((self.hidden_dim), self.n_classes,bias=False)
            
        self.last_feature_dimension = self.smax_fc.in_features
    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        if self.uni_modality =="l":
            [r1,r2,r3,r4]=U
            seq_len, _, feature_dim = r1.size()
            if r1.size(0)==1:
                pass
            else:
                r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
                r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

            U_l = (r1 + r2 + r3 + r4)/4
            U_l = self.linear_l(U_l)
        
        
        
        elif self.uni_modality =="a":
            # print(U_a.size())
            # print(self.linear_a.weight.size())
            U_a = self.linear_a(U_a)
            
        elif self.uni_modality =="v":
            U_v = self.linear_v(U_v)
        
        
        if self.args.use_speaker_embedding:
            
            spk_idx = torch.argmax(qmask, dim=-1)
            spk_emb_vector = self.speaker_embeddings(spk_idx)
            if self.uni_modality=="l":
                U_l = U_l + spk_emb_vector
            elif self.uni_modality=="v":
                U_v = U_v + spk_emb_vector
            elif self.uni_modality=="a":
                U_a = U_a + spk_emb_vector
        
        
        if self.uni_modality=="a":
            U_a = self.dropout_(U_a)
            U_a = nn.ReLU()(U_a)
            emotions_a, _ = self.enc_a(U_a)
            if self.using_MHA:
                emotions_a = self.attention_a(emotions_a,emotions_a)
                # print("오디오 MHA 계산함")

            
            if self.stablizing:
                emotions_a = self.ln_a(self.dropout_(emotions_a))
               
            else:
                emotions_a = self.dropout_(emotions_a)
            emotions_feat = emotions_a
                

        elif self.uni_modality=="v":
            U_v = self.dropout_(U_v)
            U_v = nn.ReLU()(U_v)
            emotions_v, _ = self.enc_v(U_v)
            if self.using_MHA:
                emotions_v = self.attention_v(emotions_v,emotions_v)
                # print("visual MHA 계산함")
            if self.stablizing:
                emotions_v = self.ln_v(self.dropout_(emotions_v))
            else:
                emotions_v = self.dropout_(emotions_v)
            emotions_feat = emotions_v
        

        # Language
        elif self.uni_modality=="l":
            U_l = self.dropout_(U_l)
            U_l = nn.ReLU()(U_l)
            emotions_l, _ = self.enc_l(U_l)
            if self.using_MHA:
                emotions_l = self.attention_l(emotions_l,emotions_l)
                # print("text MHA 계산함")
            if self.stablizing:
                emotions_l = self.ln_l(self.dropout_(emotions_l))
            else:
                emotions_l = self.dropout_(emotions_l)
            emotions_feat = emotions_l
            
        
        
        emotions_feat = simple_batch_graphify(emotions_feat, seq_lengths, self.no_cuda)
        
        if self.using_graph:
            emotions_feat = self.gcn(emotions_feat, seq_lengths, qmask, epoch)
        
        
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        logits = self.smax_fc(emotions_feat)
        
        return logits
