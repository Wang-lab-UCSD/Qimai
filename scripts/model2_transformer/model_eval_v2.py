import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
import torch.utils.data as Data
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc, precision_recall_curve, matthews_corrcoef

import numpy as np
import pandas as pd
import random
import os, gc
import glob
import time, timeit
import math
import GPUtil
import copy
import pickle
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from models import *
# from models_attention import *
# from models_CR import *
# from models_attention_v2 import *
# from models_attention_v3 import *



# ################# models ####################
# class LayerNorm(nn.Module):
#     def __init__(self, hid_dim, variance_epsilon=1e-12):

#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(hid_dim))
#         self.beta = nn.Parameter(torch.zeros(hid_dim))
#         self.variance_epsilon = variance_epsilon

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
        
#         out = (x - mean) / (std + self.variance_epsilon)
#         out = self.gamma * out + self.beta
#         return out
    
# class SelfAttention(nn.Module):
#     def __init__(self, hid_dim, n_heads, dropout, device):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.n_heads = n_heads

#         assert hid_dim % n_heads == 0

#         self.w_q = nn.Linear(hid_dim, hid_dim)
#         self.w_k = nn.Linear(hid_dim, hid_dim)
#         self.w_v = nn.Linear(hid_dim, hid_dim)

#         self.fc = nn.Linear(hid_dim, hid_dim)

#         self.do = nn.Dropout(dropout)

#         self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

#     def forward(self, query, key, value, mask=None):
#         bsz = query.shape[0]

#         # query = key = value [batch size, sent len, hid dim]

#         Q = self.w_q(query)
#         K = self.w_k(key)
#         V = self.w_v(value)

#         # Q, K, V = [batch size, sent len, hid dim]

#         Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
#         K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
#         V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

#         # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
#         # Q = [batch size, n heads, sent len_q, hid dim // n heads]
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

#         # energy = [batch size, n heads, sent len_Q, sent len_K]
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, -1e10)

#         attention = self.do(F.softmax(energy, dim=-1))

#         # attention = [batch size, n heads, sent len_Q, sent len_K]

#         x = torch.matmul(attention, V)

#         # x = [batch size, n heads, sent len_Q, hid dim // n heads]

#         x = x.permute(0, 2, 1, 3).contiguous()

#         # x = [batch size, sent len_Q, n heads, hid dim // n heads]

#         x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

#         # x = [batch size, src sent len_Q, hid dim]

#         x = self.fc(x)

#         # x = [batch size, sent len_Q, hid dim]

#         return x


# class PositionwiseFeedforward(nn.Module):
#     def __init__(self, hid_dim, pf_dim, dropout):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.pf_dim = pf_dim

#         self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
#         self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

#         self.do = nn.Dropout(dropout)

#     def forward(self, x):
#         # x = [batch size, sent len, hid dim]

#         x = x.permute(0, 2, 1)
#         # x = [batch size, hid dim, sent len]

#         x = self.do(F.relu(self.fc_1(x)))
#         # x = [batch size, pf dim, sent len]

#         x = self.fc_2(x)
#         # x = [batch size, hid dim, sent len]

#         x = x.permute(0, 2, 1)
#         # x = [batch size, sent len, hid dim]

#         return x
    
# class EncoderLayer(nn.Module):
#     def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
#         super().__init__()
#         self.ln1 = LayerNorm(hid_dim)
#         self.ln2 = LayerNorm(hid_dim)
        
#         self.do1 = nn.Dropout(dropout)
#         self.do2 = nn.Dropout(dropout)
        
#         self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
#         self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)

        
#     def forward(self, trg, trg_mask=None):
#         # trg = [batch_size, dna len, dna_dim]
#         # src = [batch_size, protein len, hid_dim] # encoder output
#         # trg_mask = [batch size, 1, dna sent len, dna sent len]
#         # src_mask = [batch size, 1, protein len, protein len]

#         trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))
#         trg = self.ln2(trg + self.do2(self.pf(trg)))

#         return trg

# class Encoder(nn.Module):
#     def __init__(self, dna_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
#         super().__init__()
#         # self.embed = Embedder(vocab_size, d_model)
#         # self.pe = PositionalEncoder(d_model)      
#         self.ft = nn.Linear(dna_dim, hid_dim)
#         self.n_layers = n_layers
#         self.layer = nn.ModuleList()
#         for _ in range(n_layers):
#             self.layer.append(EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device))

        
#     def forward(self, trg, trg_mask=None):
#         # trg = [batch_size, dna len, dna_dim]
        
#         trg = self.ft(trg)
#         # trg = [batch size, dna len, hid dim]

#         for layer in self.layer:
#             trg = layer(trg, trg_mask)
            
#         return trg
    
# class DecoderLayer(nn.Module):
#     def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
#         super().__init__()

#         self.ln1 = LayerNorm(hid_dim)
#         self.ln2 = LayerNorm(hid_dim)
#         self.ln3 = LayerNorm(hid_dim)
        
#         self.do1 = nn.Dropout(dropout)
#         self.do2 = nn.Dropout(dropout)
#         self.do3 = nn.Dropout(dropout)
        
#         self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
#         self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
#         self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        

#     def forward(self, trg, src, trg_mask=None, cross_attn_mask=None):
#         # trg = [batch_size, dna len, dna_dim]
#         # src = [batch_size, protein len, hid_dim] 
#         # trg_mask = [batch size, 1, dna sent len, dna sent len]
#         # cross_attn_mask = [batch, 1, dna sent len, protein len]

#         trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))
#         trg = self.ln2(trg + self.do2(self.ea(trg, src, src, cross_attn_mask)))
#         trg = self.ln3(trg + self.do3(self.pf(trg)))

#         return trg


# class Decoder(nn.Module):
#     """ dna feature extraction."""
#     def __init__(self, protein_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
#         super().__init__()
#         self.ln = LayerNorm(hid_dim)
#         self.output_dim = protein_dim
#         self.hid_dim = hid_dim
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.pf_dim = pf_dim
#         self.dropout = dropout
#         self.device = device
#         self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
#         self.layer = nn.ModuleList()
#         for _ in range(n_layers):
#             self.layer.append(DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device))

#         self.ft = nn.Linear(hid_dim, hid_dim)
#         self.output = nn.Sequential(
#             nn.Linear(self.hid_dim, 256),
#             nn.ReLU(True),
# #             nn.Dropout(p=self.dropout),
            
#             nn.Linear(256, 64),
#             nn.ReLU(True),
# #             nn.Dropout(p=self.dropout),
            
#             nn.Linear(64, 32),
#             nn.ReLU(True),
# #             nn.Dropout(p=self.dropout),
            
#             #output layer
#             nn.Linear(32, 1)
#         )
        

#     def forward(self, trg, src, trg_mask=None,cross_attn_mask=None):
#         # trg = [batch_size, protein len, hid_dim] 
#         # src = [batch_size, dna len, dna_dim]
#         # trg_mask = [batch size, 1, protein len, protein len]
#         # cross_attn_mask = [batch, 1, protein len, dna len]        
#         trg = self.ft(trg)

#         # trg = [batch size, protein len, hid dim]

#         for layer in self.layer:
#             trg = layer(trg, src, trg_mask, cross_attn_mask)

#         # trg = [batch size, protein len, hid dim]

# #         label, _ = torch.max(trg, dim=1)
#         trg_mask_2d = trg_mask[:,0,:,0]
#         label = torch.sum(trg*trg_mask_2d[:,:,None], dim=1)/trg_mask_2d.sum(dim=1, keepdims=True)
#         label = label.unsqueeze(1)
#         label = self.output(label)
        
#         return label


# class Predictor(nn.Module):
#     def __init__(self, **config):
#         super().__init__()

#         self.protein_dim = config['protein_dim']
#         self.dna_dim = config['dna_dim']
#         self.hid_dim = config['hid_dim']
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.pf_dim = config['pf_dim']
#         self.dropout = config['dropout']
#         self.batch_size = config['batch_size']
#         self.device = config['device']
        
#         self.encoder1 = Encoder(self.dna_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device)
#         self.encoder2 = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device)
#         self.decoder = Decoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device)



#     def forward(self, dna, protein, dna_mask, protein_mask, cross_attn_mask):
#         # torch.tensor
#         # dna = [batch, max_d, dna_dim]
#         # protein = [batch, max_p, protein_dim]
#         # dna_mask = [batch, 1, max_d, max_d]
#         # protein_mask = [batch, 1, max_p, max_p]
#         # cross_attn_mask = [batch, 1, max_p, max_d]

        
#         enc_src = self.encoder1(dna, dna_mask)
#         # enc_src = [batch size, dna len, hid_dim]
                
#         enc_trg = self.encoder2(protein, protein_mask)
#         # enc_trg = [batch size ,pro len, hid_dim]

#         out = self.decoder(enc_trg, enc_src, protein_mask, cross_attn_mask)

#         return out



# def predict(data_iter,model,use_cuda):
#     with torch.set_grad_enabled(False):
#         model.eval()
#         total_loss, count = 0,0
#         y_label, y_pred = [], []
#         for i, (d, p, d_mask, p_mask, label) in enumerate(data_iter):
#             dna_mask = d_mask.unsqueeze(1).unsqueeze(2) # dna_mask = [batch, 1, 1, max_d]
#             protein_mask = p_mask.unsqueeze(1).unsqueeze(3)    # protein_mask = [batch, 1, max_p, 1]
#             cross_attn_mask = torch.matmul(protein_mask, dna_mask)  # cross_attn_mask = [batch, 1, max_p, max_d]

#             dna_mask = torch.matmul(dna_mask.permute(0,1,3,2), dna_mask) # dna_mask = [batch, 1, max_d, max_d]
#             protein_mask = torch.matmul(protein_mask, protein_mask.permute(0,1,3,2)) # protein_mask = [batch, 1, max_p, max_p]

#             label = Variable(torch.from_numpy(np.array(label)).float())

#             if use_cuda:
#                 d = d.cuda()
#                 p = p.cuda()
#                 cross_attn_mask = cross_attn_mask.cuda()
#                 dna_mask = dna_mask.cuda()
#                 protein_mask = protein_mask.cuda()
#                 label = label.cuda()  

#             score = model(d, p, dna_mask, protein_mask, cross_attn_mask)
#             m = torch.nn.Sigmoid()
#             logits = torch.squeeze(m(score))

#             loss_fct = torch.nn.BCELoss()   
#             loss = loss_fct(logits, label) 

#             total_loss += loss
#             count += 1
#     #         print(loss)

#             y_label.extend(label.to('cpu').data.numpy())
#             y_pred.extend(logits.to('cpu').data.numpy())

#         try:

#             ROC = roc_auc_score(y_label, y_pred)
#             PRC = average_precision_score(y_label, y_pred)

#             fpr, tpr, thresholds = roc_curve(y_label, y_pred)

#             precision = tpr / (tpr + fpr)

#             f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

#             # get the the threshold
#             J = tpr - fpr
#             ix = np.argmax(J)
#             # thred_optim = thresholds[5:][np.argmax(f1[5:])]
#             # thred_optim = 0.5
#             thred_optim = thresholds[ix]

#             # y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
#             y_pred_s = [1 if i>thred_optim else 0 for i in y_pred]

#             auc_k = metrics.auc(fpr, tpr)
#             cm1 = confusion_matrix(y_label, y_pred_s)
#             total1=sum(sum(cm1))
#             #####from confusion matrix calculate accuracy
#             accuracy1=(cm1[0,0]+cm1[1,1])/total1
#             sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
#             specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

#             F1 = f1_score(y_label, y_pred_s)
#             recall = recall_score(y_label, y_pred_s)        
#             MCC = matthews_corrcoef(y_label, y_pred_s)

#             return accuracy1, ROC, PRC, MCC, recall, F1, total_loss / count, y_label, y_pred    
        
#         except ValueError:
#             return 0, 0, 0, 0, 0, 0, total_loss / count, y_label, y_pred

# def predict_s(data_iter,model,use_cuda):
#     with torch.set_grad_enabled(False):
#         model.eval()
#         total_loss, count = 0,0
#         y_label, y_pred, p_start = [], [], []
#         for i, (d, p, d_mask, p_mask, label, p_start_idx) in enumerate(data_iter):
#             dna_mask = d_mask.unsqueeze(1).unsqueeze(2) # dna_mask = [batch, 1, 1, max_d]
#             protein_mask = p_mask.unsqueeze(1).unsqueeze(3)    # protein_mask = [batch, 1, max_p, 1]
#             cross_attn_mask = torch.matmul(protein_mask, dna_mask)  # cross_attn_mask = [batch, 1, max_p, max_d]

#             dna_mask = torch.matmul(dna_mask.permute(0,1,3,2), dna_mask) # dna_mask = [batch, 1, max_d, max_d]
#             protein_mask = torch.matmul(protein_mask, protein_mask.permute(0,1,3,2)) # protein_mask = [batch, 1, max_p, max_p]

#             label = Variable(torch.from_numpy(np.array(label)).float())

#             if use_cuda:
#                 d = d.cuda()
#                 p = p.cuda()
#                 cross_attn_mask = cross_attn_mask.cuda()
#                 dna_mask = dna_mask.cuda()
#                 protein_mask = protein_mask.cuda()
#                 label = label.cuda()  

#             score = model(d, p, dna_mask, protein_mask, cross_attn_mask)
#             m = torch.nn.Sigmoid()
#             logits = torch.squeeze(m(score))

#             loss_fct = torch.nn.BCELoss()   
#             loss = loss_fct(logits, label) 

#             total_loss += loss
#             count += 1
#     #         print(loss)

#             y_label.extend(label.to('cpu').data.numpy())
#             y_pred.extend(logits.to('cpu').data.numpy())
#             p_start.extend(p_start_idx.to('cpu').data.numpy())
            
#         try:

#             ROC = roc_auc_score(y_label, y_pred)
#             PRC = average_precision_score(y_label, y_pred)

#             fpr, tpr, thresholds = roc_curve(y_label, y_pred)

#             precision = tpr / (tpr + fpr)

#             f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

#             # get the the threshold
#             J = tpr - fpr
#             ix = np.argmax(J)
#             # thred_optim = thresholds[5:][np.argmax(f1[5:])]
#             # thred_optim = 0.5
#             thred_optim = thresholds[ix]

#             # y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
#             y_pred_s = [1 if i>thred_optim else 0 for i in y_pred]

#             auc_k = metrics.auc(fpr, tpr)
#             cm1 = confusion_matrix(y_label, y_pred_s)
#             total1=sum(sum(cm1))
#             #####from confusion matrix calculate accuracy
#             accuracy1=(cm1[0,0]+cm1[1,1])/total1
#             sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
#             specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

#             F1 = f1_score(y_label, y_pred_s)
#             recall = recall_score(y_label, y_pred_s)        
#             MCC = matthews_corrcoef(y_label, y_pred_s)

#             return accuracy1, ROC, PRC, MCC, recall, F1, total_loss / count, y_label, y_pred, p_start    
        
#         except ValueError:
#             return 0, 0, 0, 0, 0, 0, total_loss / count, y_label, y_pred, p_start
          
######################### end of models #############################

####### set up path ###################
# TF = "FOS"
# TF = ["FOS","CTCF","RUNX3","AP2A","RAD21"]


# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer_1_segmentation_100fold,num_files=10,max_dna=512,max_protein=50.pt"
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer_1_segmentation_10fold,num_files=10,max_dna=512,max_protein=50.pt"
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer_1,num_files=1,max_dna=512,max_protein=50.pt"

# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=512.pt"
# model_path = "v10-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=512.pt"
# model_path = "v10-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=768.pt"
# model_path = "v11-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=512.pt"
# model_path = "v11_consecutive-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=512.pt"
# model_path = 'lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=HepG2_valid_balanced,num_files=22,max_dna=512,max_protein=512.pt'
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=768.pt"
# model_path = 'lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=MCF-7_valid_balanced,num_files=8,max_dna=512,max_protein=512.pt'
# model_path = 'model_CR_v2-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=512.pt'

# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=MCF-7_valid,num_files=32,max_dna=512,max_protein=50.pt"
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=MCF-7_valid,num_files=32,max_dna=512,max_protein=512.pt"
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train,num_files=6,max_dna=512,max_protein=512.pt"

# model_path = "lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=MCF-7_train,num_files=428,max_dna=512,max_protein=512.pt"
# model_path = 'v13-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=17,max_dna=512,max_protein=512.pt'
# model_path = 'v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=18,max_dna=512,max_protein=768.pt'
# model_path = 'model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=240,max_dna=512,max_protein=768.pt'
# model_path = 'main_v6_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=240,max_dna=512,max_protein=768.pt'
# model_path = 'model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=859,max_dna=512,max_protein=768.pt'
# model_path = 'model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=746,max_dna=512,max_protein=768.pt'
# model_path = 'ChIP_690_full_model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=440,max_dna=512,max_protein=768.pt'
model_path = 'main_v13_deepsea_ChIP_690-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=746,max_dna=512,max_protein=768.pt'
# model_path = 'model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=12,batch=16,input=train,num_files=746,max_dna=512,max_protein=768.pt'

# list all evaluation data
# eval_data = '100_per_exp_max_512bp_train_6mer'
# eval_data = '100_per_exp_max_512bp_train_6mer_1_segmentation_10fold_1'
# eval_data = '100_per_exp_max_512bp_train_6mer_1_segmentation_100fold_1'
# eval_data = "MCF-7_valid_6mer"
# eval_data = '100_per_exp_max_512bp_test_6mer'
# eval_data = 'HepG2_valid_balanced_6mer_6'
# eval_data = "MCF-7_valid_balanced"
# eval_data = 'all_valid_6mer'
# eval_data = '100_per_exp_max_512bp'
# eval_data = '100_per_exp_max_512bp_test_6mer'
# eval_data = 'test_6mer'
eval_data = '*_test'
# eval_data = '*_all_test'

# dataset = 'Encode3'
dataset = 'ChIP_690'

# dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/embeddings/'
# files = glob.glob(dataFolder+eval_data+'*pkl')
# files = glob.glob('/new-stg/home/cong/DPI/dataset/**/'+eval_data+'*pkl', recursive=True)
files = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+eval_data+'*pkl', recursive=True)
files.sort()

# f1 = glob.glob('/new-stg/home/cong/DPI/dataset/Encode3/**/'+eval_data+'_[0-9].pkl', recursive=True)
# f2 = glob.glob('/new-stg/home/cong/DPI/dataset/ChIP_690/**/'+eval_data+'_[0-9].pkl', recursive=True)
# files =f1+f2

print(files)
print(len(files))


def config():
    config = {}
    config['batch_size'] = 16
    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
    
#     config['max_protein_seq'] = 512
    config['max_protein_seq'] = 768
    config["warmup"]  = 5000
    config['files_per_split'] = 1
    config['hid_dim'] = 240
    config['dropout'] = 0.1
    config['lr'] = 1e-2
    config['pf_dim'] = 2048
    config['n_layers'] = 12
    config['n_heads'] = 6
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['return_p_start_idx'] = True
    
    return config


def init_model(config):

    model = Predictor(**config)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# load DNABERT and alphaFold embeddings
def unify_dim_embedding(x,crop_size,return_start_index=False):
    """
    crops/pads embedding to unify the dimention implementing AlphaFold's cropping strategy

    :param x: protein or dna embedding (seq_length,) or (seq_length,emb_size)
    :type x: numpy.ndarray
    :param crop_size: crop size, either max_p or max_d
    :type crop_size: int
    :return x: unified embedding (crop_size,)
    :type x: numpy.ndarray

    """
    pad_token_index = 0
    seq_length = x.shape[0]
    start_pos = 0 # set default 
    if seq_length < crop_size:
        input_mask = ([1] * seq_length) + ([0] * (crop_size - seq_length))
        if x.ndim == 1:
            x = torch.from_numpy(np.pad(x, (0, crop_size - seq_length), 'constant', constant_values = pad_token_index))
        elif x.ndim == 2:
            x = torch.from_numpy(np.pad(x, ((0, crop_size - seq_length),(0,0)), 'constant', constant_values = pad_token_index))
    else:
        start_pos = random.randint(0,seq_length-crop_size)
        x = x[start_pos:start_pos+crop_size]
        if isinstance(x, np.ndarray):
#         if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        input_mask = [1] * crop_size
    if return_start_index:
        return x, np.asarray(input_mask), start_pos
    else:
        return x, np.asarray(input_mask)
    

class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti, max_d, max_p, return_start_index):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.max_d = max_d
        self.max_p = max_p
        self.return_start_index = return_start_index
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        d = self.df.iloc[index]['dna_embedding']
        d_v, input_mask_d = unify_dim_embedding(d,self.max_d)
        
        p = self.df.iloc[index]['protein_embedding']
        
        if return_start_index:
            p_v, input_mask_p, p_start_idx = unify_dim_embedding(p,self.max_p,return_start_index=return_start_index)        
            y = self.labels[index]
            return d_v, p_v, input_mask_d, input_mask_p, y, p_start_idx
        else:
            p_v, input_mask_p = unify_dim_embedding(p,self.max_p,return_start_index=return_start_index)
            y = self.labels[index]
            return d_v, p_v, input_mask_d, input_mask_p, y
    


def create_datasets(batch_size, df):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0, 
          'drop_last': True,
          'pin_memory': True}
        
    print("dataset size: ", df.shape[0])
    
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p, return_start_index)
    generator = data.DataLoader(dataset, **params)
    
    return generator, len(dataset)

if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    use_cuda = torch.cuda.is_available()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
        
    """ create config"""
    config = config()
    print(config)

    device = config['device']
    batch = config['batch_size']

    max_d = config['max_dna_seq']
    max_p = config['max_protein_seq']
    files_per_split = config['files_per_split'] 
    return_start_index = config['return_p_start_idx']
        
#     """Load preprocessed data."""
#     dataset, dataset_size = create_datasets(batch, eval_data)


    """ Load trained model"""
    model = init_model(config)
    file_model = 'output/model/' + model_path
    model.load_state_dict(torch.load(file_model))
    model.to(device)


    """Output files."""
    file_AUCs = 'output/result/eval--' + dataset+'-'+eval_data + "," + model_path.replace('pt','txt')
    file_labels = 'output/result/eval-label--' + dataset + '-' + eval_data + model_path.replace('pt','txt')
    
    AUC = ('Split\tTotal_files\tProtein\tTime(sec)\tLoss_val\taccuracy_dev\tROC_dev\tPRC_dev\tMCC_dev\trecall_dev\tF1_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    
    """create output folder for evaluation results"""
    output_folder = "output/attention_analysis/model="+model_path.replace(".pt","")
    Path(output_folder).mkdir(exist_ok=True)
    

    """Start evaluating"""
    print('Evaluating...')
    start = timeit.default_timer()
    num_split = math.ceil(len(files)/files_per_split)
    for i in range(0, num_split):
        chosen_files = files[i*files_per_split: min((i+1)*files_per_split, len(files))]
        li = []
        for filename in chosen_files:
            print(filename)
            df = pd.read_pickle(filename)
            li.append(df)
        frame = pd.concat(li, axis=0, ignore_index=True)
        filter = frame['dna'].str.contains('N')
        frame = frame[~filter]
        print(len(frame))
#         frame = frame.iloc[0:10000]
#         frame = frame.sample(n = 10000)
        print(len(frame))
        del li
        """Load preprocessed data"""
        for x, y in frame.groupby('protein'):
            ### only test specific TF
#             if x == TF:
#             if x in TF:
            print("testing TF: "+x)
            print("number in dataset: "+str(y.shape[0]))

            if y.shape[0] >= batch:
#                 # set upper limit for number of examples (optional)
#                 upper = 1000 
#                 if y.shape[0] >= upper:
#                     y = y.sample(n=upper)
#                     print('sampling '+str(upper)+' examples.')
                    
                dataset, dataset_size = create_datasets(batch, y)

                print(f"Split {i+1} Validation ====", flush=True)
                with torch.set_grad_enabled(False):
                    model.eval()
                    if return_start_index:
                        accuracy, ROC_dev, PRC_dev, MCC, recall, f1, loss_dev, y_label, y_pred, p_start = predict_s(dataset,model,use_cuda)
                    else:
                        accuracy, ROC_dev, PRC_dev, MCC, recall, f1, loss_dev, y_label, y_pred = predict(dataset,model,use_cuda)
                end = timeit.default_timer()
                time = end - start

                loss_dev = loss_dev.to('cpu').data.numpy()
            #         train_loss.append(loss_train)
            #         val_loss.append(loss_dev)
            #         val_auc.append(AUC_dev)
            #         val_accuracy.append(accuracy)
                AUCs = [i+1, (i+1)*files_per_split, x, time, loss_dev, accuracy, ROC_dev, PRC_dev, MCC, recall, f1]
                print('\t'.join(map(str, AUCs))) 

                with open(file_AUCs, 'a') as f:
                    f.write('\t'.join(map(str, AUCs)) + '\n')

                # write start info to file----------------------------------------
    #                 df = pd.DataFrame({"true": y_label, "pred": y_pred})
                if return_start_index:
#                     df = pd.DataFrame({"true": y_label, "pred": y_pred, "attention_protein":attention_protein, "attention_dna": attention_dna, "attention_cross": attention_cross, "p_start_idx": p_start})
                    df = pd.DataFrame({"true": y_label, "pred": y_pred, "p_start_idx": p_start})
                    file_labels = output_folder+'/eval-label--split'+str(i+1)+',protein=' +x+',eval=' + eval_data + ',model='+model_path.replace('pt','pkl')
    #                     file_labels = 'output/result/eval-label--split'+str(i+1)+',protein=' +x+',eval=' + 'segmentation_10fold_1' + ',model='+model_path.replace('pt','pkl')
                    df.to_pickle(file_labels)                
            else:
                print(x+" only has "+str(y.shape[0])+" samples.")

        
    print('Finished evaluating...')    

