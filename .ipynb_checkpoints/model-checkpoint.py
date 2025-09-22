


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math

def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True, args = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
        self.args = args
    
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
        if self.args.focal_prob == 'prob':
            prob = torch.exp(log_p)    
        elif self.args.focal_prob == 'log_prob':
            prob = log_p
        else:
            raise ValueError("Unknown focal_prob type")
        
        pt = label_onehot * prob
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

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

        # 输出线性变换（残差）
        output = img_features + self.output_proj(attended_features)  
        return output


class Model(nn.Module):

    def __init__(self, 
                 D_m,
                 D_g,  
                 n_speakers,
                 n_classes=7, 
                 dropout=0.5,
                 no_cuda=False, 
                 use_residue=True,
                 D_m_v=512,
                 D_m_a=100,
                 dataset='IEMOCAP',
                  args=None):
        
        super(Model, self).__init__()
        self.args = args
        self.no_cuda = no_cuda
        self.n_speakers = n_speakers
        self.dropout = dropout
        self.use_residue = use_residue
        self.D_g = D_g
        self.multi_modal = True
        self.dataset = dataset
        self.stablizing = args.stablizing
        
        # BN for four textual streams
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)
        if self.args.use_speaker_embedding:
            self.speaker_embeddings = nn.Embedding(n_speakers,  D_g)
        # Encoders per modality
        hidden_a = D_g
        self.linear_a = nn.Linear(D_m_a, hidden_a)
        self.enc_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
           

        hidden_v = D_g
        self.linear_v = nn.Linear(D_m_v, hidden_v)
        self.enc_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
           

        hidden_l = D_g
        self.linear_l = nn.Linear(D_m, hidden_l)
        self.enc_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
           

        # Cross-modal attentions (target <- source)
        self.align_a_a = MultiHeadCrossModalAttention(D_g, D_g, D_g, self.args.num_heads_audio) 
        self.align_v_v = MultiHeadCrossModalAttention(D_g, D_g, D_g, self.args.num_heads_visual)
        self.align_l_l = MultiHeadCrossModalAttention(D_g, D_g, D_g, self.args.num_heads_text)
        
        # 결합 후 안정화용
        self.ln_a = nn.LayerNorm(D_g)
        self.ln_v = nn.LayerNorm(D_g)
        self.ln_l = nn.LayerNorm(D_g)
        self.dropout_ = nn.Dropout(self.dropout)


        self.num_modals = 3
        if self.use_residue:
            self.smax_fc = nn.Linear((2*D_g)*self.num_modals, n_classes)
        else:
            self.smax_fc = nn.Linear((D_g)*self.num_modals, n_classes)
            self.last_feature_dimension = (D_g)*self.num_modals

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
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
        
        original_emotions_a, original_emotions_v, original_emotions_l = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda), simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda), simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            # ----- Cross-modal fusion with gating (softmax over 3 terms per modality)
        if self.args.using_crossmodal_attention:
            
            a0, v0, l0 = emotions_a, emotions_v, emotions_l

           
            emotions_a = self.align_a_a(a0, a0)
            emotions_v = self.align_v_v(v0, v0)
            emotions_l = self.align_l_l(l0, l0)

           
            
        else:
            pass

        # 결합 후 안정화 (모든 모드에 동일 적용 권장)
        if self.stablizing:
            emotions_a = self.ln_a(self.dropout_(emotions_a))
            emotions_v = self.ln_v(self.dropout_(emotions_v))
            emotions_l = self.ln_l(self.dropout_(emotions_l))
        else:
            emotions_a = self.dropout_(emotions_a)
            emotions_v = self.dropout_(emotions_v)
            emotions_l = self.dropout_(emotions_l)
        # ----- end fusion
       
        # 그냥 결합
        
        emotions_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
        emotions_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
        emotions_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
        # print(emotions_a.size(), emotions_v.size(), emotions_l.size())
        if self.use_residue:
            emotions_feat = torch.cat([emotions_a, emotions_v, emotions_l,original_emotions_a,original_emotions_v, original_emotions_l], dim=-1)
        else:
            emotions_feat = torch.cat([emotions_a, emotions_v, emotions_l], dim=-1)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        logits = self.smax_fc(emotions_feat)
        
        return logits
