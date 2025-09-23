

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math

def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad
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
                  args=None):
        
        super(Model, self).__init__()
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
           

        # 결합 후 안정화용
        self.ln_a = nn.LayerNorm(self.hidden_dim)
        self.ln_v = nn.LayerNorm(self.hidden_dim)
        self.ln_l = nn.LayerNorm(self.hidden_dim)
        self.dropout_ = nn.Dropout(self.dropout)


        self.num_modals = 3

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
                args=args
                )

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        logits_modal = {}
        logits_uni_modal_student ={}
        logits_MKD_teacher = {}
        logits_MRL_feature = {}
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
        

        # 결합 후 안정화 (모든 모드에 동일 적용 권장)
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

        if self.calib:
            logits_modal = self.modal_comb_forward(emotions_a, emotions_v, emotions_l)
        
        if self.MKD:
            logits_uni_modal_student['a']=self.student_a(emotions_a)
            logits_uni_modal_student['v']=self.student_v(emotions_v)
            logits_uni_modal_student['l']=self.student_l(emotions_l)
        
        emotions_feature = torch.cat([emotions_a, emotions_v, emotions_l], dim=-1)

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
        return logits, logits_uni_modal_student, logits_MKD_teacher, logits_MRL_feature, logits_modal
    
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
                 args=None):
        
        super(Unimodal_MODEL, self).__init__()
        self.n_classes = n_classes
        self.args = args
        self.no_cuda = no_cuda
        self.n_speakers = n_speakers
        self.dropout = dropout

        self.hidden_dim = hidden_dim        
        self.dataset = dataset
        self.stablizing = args.stablizing
        
        
        
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
            self.ln_l = nn.LayerNorm(self.hidden_dim)

        
        elif self.uni_modality == "a":
            hidden_a = self.hidden_dim
            self.linear_a = nn.Linear(D_m_a, hidden_a)
            self.enc_a = nn.LSTM(input_size=hidden_a, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)
            self.ln_a = nn.LayerNorm(self.hidden_dim)

        elif self.uni_modality == "v":
            hidden_v = self.hidden_dim
            self.linear_v = nn.Linear(D_m_v, hidden_v)
            self.enc_v = nn.LSTM(input_size=hidden_v, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, dropout=dropout)
            self.ln_v = nn.LayerNorm(self.hidden_dim)


        
        # 결합 후 안정화용
       
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
            if self.stablizing:
                emotions_a = self.ln_a(self.dropout_(emotions_a))
               
            else:
                emotions_a = self.dropout_(emotions_a)
            emotions_feat = emotions_a
                

        elif self.uni_modality=="v":
            U_v = self.dropout_(U_v)
            U_v = nn.ReLU()(U_v)
            emotions_v, _ = self.enc_v(U_v)
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
            if self.stablizing:
                emotions_l = self.ln_l(self.dropout_(emotions_l))
            else:
                emotions_l = self.dropout_(emotions_l)
            emotions_feat = emotions_l
            
        
        
        emotions_feat = simple_batch_graphify(emotions_feat, seq_lengths, self.no_cuda)
        
        
        
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        logits = self.smax_fc(emotions_feat)
        
        return logits
