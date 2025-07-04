#### define transformer modules: LayerNorm, SelfAttention, PositionwiseFeedforward, EncoderLayer, Encoder, DecoderLayer
#### define relational graph modeules: get_degree_mat, get_laplace_mat, GCNConv, RGmodule
#### define utility function: evaluate

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import copy
import numpy as np
import time
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc, precision_recall_curve, matthews_corrcoef

### utils function ####
def read_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.rstrip('\n'))
    return lines

### transformer modules####
class LayerNorm(nn.Module):
    def __init__(self, hid_dim, variance_epsilon=1e-12):

        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hid_dim))
        self.beta = nn.Parameter(torch.zeros(hid_dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        out = (x - mean) / (std + self.variance_epsilon)
        out = self.gamma * out + self.beta
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, return_attention=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device
        self.return_attention = return_attention

        assert self.hid_dim % self.n_heads == 0

        self.w_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.w_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.w_v = nn.Linear(self.hid_dim, self.hid_dim)

        self.fc = nn.Linear(self.hid_dim, self.hid_dim)

        self.do = nn.Dropout(self.dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(self.device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            mask_value = -1e4 if energy.dtype == torch.float16 else -1e10
            energy = energy.masked_fill(mask == 0, mask_value)
        if torch.isnan(energy).any():
            print("NaN detected in protein encoder layer: energy")
        attention = self.do(F.softmax(energy, dim=-1))
        if torch.isnan(attention).any():
            print("NaN detected in protein encoder layer: attention")
        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        if torch.isnan(x).any():
            print("NaN detected in protein encoder layer: x")
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        if self.return_attention:
            return x, attention
        else:
            return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]

        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, return_attention):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.return_attention = return_attention
        
        self.ln1 = LayerNorm(self.hid_dim)
        self.ln2 = LayerNorm(self.hid_dim)
        
        self.do1 = nn.Dropout(self.dropout)
        self.do2 = nn.Dropout(self.dropout)
        
        self.sa = SelfAttention(self.hid_dim, self.n_heads, self.dropout, self.device, self.return_attention)
#         self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(self.hid_dim, self.pf_dim, self.dropout)

        
    def forward(self, trg, trg_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, 1, dna sent len, dna sent len]
        # src_mask = [batch size, 1, protein len, protein len]

        if self.return_attention: 
            trg_1 = trg
            trg, attention = self.sa(trg, trg, trg, trg_mask)
            trg = self.ln1(trg_1 + self.do1(trg))

            trg = self.ln2(trg + self.do2(self.pf(trg)))

            return trg, attention
        else:
            trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))
            # if torch.isnan(trg).any():
            #     print("NaN detected in protein encoder layer: self attention")
            trg = self.ln2(trg + self.do2(self.pf(trg)))
            # if torch.isnan(trg).any():
            #     print("NaN detected in protein encoder layer: feed forward")
            return trg

class Encoder(nn.Module):
    def __init__(self, dna_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, return_attention=False):
        super().__init__()
        self.dna_dim = dna_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.return_attention = return_attention
        
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)      
        self.ft = nn.Linear(self.dna_dim, self.hid_dim)
        self.layer = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layer.append(EncoderLayer(self.hid_dim, self.n_heads, self.pf_dim, self.dropout, self.device, self.return_attention))

        
    def forward(self, trg, trg_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        
        trg = self.ft(trg)
        # trg = [batch size, dna len, hid dim]

        if self.return_attention:
            for layer in self.layer:
                trg, attention = layer(trg, trg_mask)
            
            return trg, attention
        else:
            for layer in self.layer:
                trg = layer(trg, trg_mask)
                if torch.isnan(trg).any():
                    print(f"NaN detected in protein encoder layer {layer}")
                    break

            return trg
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, return_attention):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.return_attention = return_attention
        
        self.ln1 = LayerNorm(self.hid_dim)
        self.ln2 = LayerNorm(self.hid_dim)
        self.ln3 = LayerNorm(self.hid_dim)
        
        self.do1 = nn.Dropout(self.dropout)
        self.do2 = nn.Dropout(self.dropout)
        self.do3 = nn.Dropout(self.dropout)
        
        self.sa = SelfAttention(self.hid_dim, self.n_heads, self.dropout, self.device) # this could be additional attention weight to consider in interpretation
        self.ea = SelfAttention(self.hid_dim, self.n_heads, self.dropout, self.device, self.return_attention)
        self.pf = PositionwiseFeedforward(self.hid_dim, self.pf_dim, self.dropout)
        

    def forward(self, trg, src, trg_mask=None, cross_attn_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        # src = [batch_size, protein len, hid_dim] 
        # trg_mask = [batch size, 1, dna sent len, dna sent len]
        # cross_attn_mask = [batch, 1, dna sent len, protein len]

        if self.return_attention:
            trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))

            trg_1 = trg
            trg, attention = self.ea(trg, src, src, cross_attn_mask)
            trg = self.ln2(trg_1 + self.do2(trg))

            trg = self.ln3(trg + self.do3(self.pf(trg)))

            return trg, attention
        else:
            trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))
            trg = self.ln2(trg + self.do2(self.ea(trg, src, src, cross_attn_mask)))
            trg = self.ln3(trg + self.do3(self.pf(trg)))

            return trg


#### Relation graph module #####
def get_degree_mat(adj_mat, pow=1, degree_version='v1'):
    degree_mat = torch.eye(adj_mat.size()[0]).to(adj_mat.device)

    if degree_version == 'v1':
        degree_list = torch.sum((adj_mat > 0), dim=1).float()
    elif degree_version == 'v2':
        # adj_mat_hat = adj_mat.data
        # adj_mat_hat[adj_mat_hat < 0] = 0
        adj_mat_hat = F.relu(adj_mat)
        degree_list = torch.sum(adj_mat_hat, dim=1).float()
    elif degree_version == 'v3':
        degree_list = torch.sum(adj_mat, dim=1).float()
        degree_list = F.relu(degree_list)
    else:
        exit('error degree_version ' + degree_version)
    degree_list = torch.pow(degree_list, pow)
    degree_mat = degree_mat * degree_list
    # degree_mat = torch.pow(degree_mat, pow)
    # degree_mat[degree_mat == float("Inf")] = 0
    # degree_mat.requires_grad = False
    # print('degree_mat = ', degree_mat)
    return degree_mat


def get_laplace_mat(adj_mat, type='sym', add_i=False, degree_version='v2'):
    if type == 'sym':
        # Symmetric normalized Laplacian
        if add_i is True:
            adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        else:
            adj_mat_hat = adj_mat
        # adj_mat_hat = adj_mat_hat[adj_mat_hat > 0]
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5, degree_version=degree_version)
        # print(degree_mat_hat.dtype, adj_mat_hat.dtype)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        # print(laplace_mat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)
        return laplace_mat
    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        return laplace_mat

class GCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
#                  dropout=0.6,
                 bias=True
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
#         self.dropout = dropout
        self.bias = bias
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias is True:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, node_ft, adj_mat):
        laplace_mat = get_laplace_mat(adj_mat, type='sym')
        node_state = torch.mm(laplace_mat, node_ft)
        node_state = torch.mm(node_state, self.weight)
        if self.bias is not None:
            node_state = node_state + self.bias

        return node_state

    
class RGmodule(nn.Module):
    def __init__(self, protein_dim, h_dim, dropout, device):
        super(RGmodule, self).__init__()

        self.protein_dim = protein_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device

        self.protein_linear = nn.Linear(self.protein_dim, self.h_dim)
        self.share_linear = nn.Linear(self.h_dim, self.h_dim)
        self.share_gcn1 = GCNConv(self.h_dim, self.h_dim)
        self.share_gcn2 = GCNConv(self.h_dim, self.h_dim)

        self.protein_adj_trans = nn.Linear(self.h_dim, self.h_dim)

        self.cross_scale_merge = nn.Parameter(
            torch.ones(1)
        )


        self.global_linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.pred_linear = nn.Linear(self.h_dim, 1)

        self.activation = nn.ELU()
        for m in self.modules():
            self.weights_init(m)
            
        self.local_linear1 = nn.Linear(1024, 512)
        self.local_linear2 = nn.Linear(512, 512)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, protein_node_ft, protein_idx):
        '''
        :param  'protein_node_ft': FloatTensor  node_num * protein_dim(384)
                'protein_idx': LongTensor  batch
        :return: 

        '''
        protein_graph_node_num = protein_node_ft.size()[0]
        protein_res_mat = torch.zeros(protein_graph_node_num, self.h_dim).to(self.device)

        # linear layer
        protein_node_ft = self.protein_linear(protein_node_ft)
#         print(protein_node_ft.shape)
#         print(protein_res_mat.shape)
        protein_res_mat = protein_res_mat + protein_node_ft
        protein_node_ft = self.activation(protein_node_ft)
        protein_node_ft = F.dropout(protein_node_ft, p=self.dropout, training=self.training)

        # generate protein adj
        protein_trans_ft = self.protein_adj_trans(protein_node_ft)
        protein_trans_ft = torch.tanh(protein_trans_ft)
        w = torch.norm(protein_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        protein_adj = torch.mm(protein_trans_ft, protein_trans_ft.t()) / w_mat

        # generate protein embedding
        protein_node_ft = self.share_gcn1(protein_node_ft, protein_adj)
        protein_res_mat = protein_res_mat + protein_node_ft

        protein_node_ft = self.activation(protein_res_mat)  # add
        protein_node_ft = F.dropout(protein_node_ft, p=self.dropout, training=self.training)
        protein_node_ft = self.share_gcn2(protein_node_ft, protein_adj)
        protein_res_mat = protein_res_mat + protein_node_ft

        protein_res_mat = self.activation(protein_res_mat)

        # get the current protein embedding
        protein_res_mat = protein_res_mat[protein_idx]

        return protein_res_mat, protein_adj


def evaluate(y_pred, y_label):
    try: 
        ROC = roc_auc_score(y_label, y_pred)
        PRC = average_precision_score(y_label, y_pred)

        # get the the threshold
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        J = tpr - fpr
        ix = np.argmax(J)
        thred_optim = thresholds[ix]

        y_pred_s = [1 if i>thred_optim else 0 for i in y_pred]
        cm1 = confusion_matrix(y_label, y_pred_s)
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_s).ravel()
        total1=sum(sum(cm1))
        print("tn|fp|fn|tp")
        print("------------")
        print('{}|{}|{}|{}'.format(tn,fp,fn,tp))
        
        #####from confusion matrix calculate accuracy
        accuracy=(tn+tp)/total1
        specificity = tn/(tn+fp) 
        sensitivity = tp/(tp+fn)

        F1 = f1_score(y_label, y_pred_s)
        MCC = matthews_corrcoef(y_label, y_pred_s)
        return accuracy, sensitivity, specificity, ROC, PRC, MCC, F1

    except ValueError:
        return 0, 0, 0, 0, 0, 0, 0




     