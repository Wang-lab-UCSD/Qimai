# protein as target
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
import seaborn as sns
from dpi_v0.models_essential import *
from dpi_v0.plot_utils import *

# input_data = 'train'
# valid_data = 'valid'
# dataset = 'ChIP_690'

input_data = 'train_min10'
valid_data = 'valid_min10'
dataset = 'Encode3and4'

# train/valid protein split
if dataset == 'ChIP_690':
#     train_pro_file = "/new-stg/home/cong/DPI/dataset/ChIP_690/mmseqs_seq_id_03_c_05_116_proteins.txt"
#     valid_pro_file = "/new-stg/home/cong/DPI/dataset/ChIP_690/mmseqs_seq_id_03_c_05_40_proteins_left_out.txt"
    # train_pro_file = "/new-stg/home/cong/DPI/dataset/ChIP_690/mmseqs_seq_id_03_c_05_random_116_proteins.txt"
    # train_pro_file = "/new-stg/home/cong/DPI/dataset/ChIP_690/C2H2_20.txt"
    train_pro_file = "/new-stg/home/cong/DPI/dataset/ChIP_690/20_bZIP_bHLH.txt"
#     train_pro_file = "/new-stg/home/cong/DPI/dataset/ChIP_690/20_from_116_protein_centers.txt"
    
elif dataset == 'Encode3and4':
#         train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_03_c_05_train_380_proteins.txt"
#         valid_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_03_c_05_valid_100_proteins.txt"
#     train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_05_c_05_train_611_proteins.txt"
#     valid_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_05_c_05_valid_224_proteins.txt"
#     train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_07_c_05_train_800_proteins.txt"
    # train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/seed101_800_proteins.txt"
    train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/proteins.txt"

#     valid_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_07_c_05_valid_35_proteins.txt"
#     train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_01_first_200_proteins.txt"
#     train_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_05_c_05_train_frac08_train_611_proteins.txt"
    # train_pro_file = "/new-stg/home/cong/DPI/dataset/all_C2H2_pros.txt"
#     valid_pro_file = "/new-stg/home/cong/DPI/dataset/Encode3and4/mmseqs_seq_id_05_c_05_valid_frac02_train_611_proteins.txt"

# tag = 'main_singletask_'+dataset+'_full_balance_by_protein'
# tag = 'main_singletask_all_'+dataset
# tag = 'main_singletask_'+dataset+'_frac_07_pros_do_on_input'
# tag = 'main_singletask_'+dataset+'_seq_id_03_c_05_pros_do_on_input'
# tag = 'main_singletask_'+dataset+'_seq_id_03_c_05'
# tag = 'main_singletask_'+dataset+'_seq_id_03_c_05_random'
# tag = 'main_singletask_'+dataset+'_C2H2_454'
# tag = 'main_singletask_'+dataset+'_seed101_800_proteins'
tag = 'main_singletask_'+dataset+'_all_847_proteins'
# tag = 'main_singletask_'+dataset+'_bZIP_bHLH_20'
# tag = 'main_singletask_'+dataset+'_diverse_20'
# tag = 'main_singletask_'+dataset+'_seq_id_05_c_05'
# tag = 'main_singletask_'+dataset+'_seq_id_05_c_05_train_frac08'
# tag = 'main_singletask_'+dataset+'_seq_id_01_first_200'
# tag = 'main_singletask_'+dataset+'_seq_id_07_c_05'
# tag = 'main_singletask_'+dataset+'_seq_id_03_c_05_left_out_valid_pros'
# tag = 'main_singletask_'+dataset+'_seq_id_05_c_05_left_out_valid_pros'
# tag = 'main_singletask_'+dataset+'_frac_07_pros'
# tag = 'main_singletask_'+dataset+'_valid_03_pros_do_on_input'
maindir = '/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'
files_dna_train = glob.glob(maindir+input_data+'/'+input_data+'_dna*pkl', recursive=True)

files_dna_train.sort()
# print('-------'+dataset+'-------')
# print('training dna embedding files: '+str(len(files_dna_train)))

files_dna_valid = glob.glob(maindir+valid_data+'/'+valid_data+'_dna*pkl', recursive=True)
files_dna_valid.sort()
# print('validation dna embedding files: '+str(len(files_dna_valid)))
 
def get_config():
    config = {}
    config['batch_size'] = 128
#     config['batch_size'] = 16

    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
#     config['n_family'] = 76
#     config['n_family'] = 65
    config['n_family'] = 0
    
    config['max_protein_seq'] = 768
#     config['max_protein_seq'] = 512

    config["warmup"]  = 10000
    config['iteration_per_split'] = 2
    config['files_per_split'] = 10
    config['valid_dna_size'] = 500

    config['hid_dim'] = 240
    config['dropout'] = 0.2
    # config['dropout'] = 0.1
    
    config['input_dropout'] = 0
#     config['lr'] = 0.2 # for all data
#     config['lr'] = 0.05 # for balanced data
    config['lr'] = 0.05
    config['pf_dim'] = 2048
    config['n_layers'] = 2
    config['n_heads'] = 6
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['return_attention'] = False
    
    return config

################# models ####################
class Decoder(nn.Module):
    """ dna feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, n_labels, return_attention=False):
        super().__init__()
        self.ln = LayerNorm(hid_dim)
        self.output_dim = protein_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.n_labels = n_labels
        self.return_attention = return_attention
        
        self.sa = SelfAttention(self.hid_dim, self.n_heads, self.dropout, self.device, self.return_attention)
        self.layer = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layer.append(DecoderLayer(self.hid_dim, self.n_heads, self.pf_dim, self.dropout, self.device, self.return_attention))

        self.ft = nn.Linear(self.hid_dim, self.hid_dim)
        self.output = nn.Sequential(
            nn.Linear(self.hid_dim, 256),
            nn.ReLU(True),
#             nn.Dropout(p=self.dropout),
            
            nn.Linear(256, 64),
            nn.ReLU(True),
#             nn.Dropout(p=self.dropout),
            
#             nn.Linear(64, 32),
#             nn.ReLU(True),
# #             nn.Dropout(p=self.dropout),
            
            #output layer
            nn.Linear(64, self.n_labels)
        )
        

    def forward(self, trg, src, trg_mask=None,cross_attn_mask=None):
        # trg = [batch_size, protein len, hid_dim] 
        # src = [batch_size, dna len, dna_dim]
        # trg_mask = [batch size, 1, protein len, protein len]
        # cross_attn_mask = [batch, 1, protein len, dna len]        
        trg = self.ft(trg)

        # trg = [batch size, protein len, hid dim]

        if self.return_attention: 
            for layer in self.layer:
                trg, attention = layer(trg, src, trg_mask, cross_attn_mask) # actually only return the last layer attention
        else:
            for layer in self.layer:
                trg = layer(trg, src, trg_mask, cross_attn_mask)

        # trg = [batch size, protein len, hid dim]

#         label, _ = torch.max(trg, dim=1)
        trg_mask_2d = trg_mask[:,0,:,0]
        label = torch.sum(trg*trg_mask_2d[:,:,None], dim=1)/trg_mask_2d.sum(dim=1, keepdims=True)
        label = label.unsqueeze(1)
        label = self.output(label)
        
        if self.return_attention: 
            return label, attention
        else:
            return label



class Predictor(nn.Module):
    def __init__(self, **config):
        super().__init__()

        self.protein_dim = config['protein_dim']
        self.dna_dim = config['dna_dim']
        self.hid_dim = config['hid_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.pf_dim = config['pf_dim']
        self.dropout = config['dropout']
        self.input_dropout = config['input_dropout']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.n_labels = config['n_family']+1
        self.return_attention = config['return_attention']

        
        self.encoder1 = Encoder(self.dna_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device, self.return_attention)
        self.encoder2 = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device, self.return_attention)
        self.decoder = Decoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device, self.n_labels, self.return_attention)
        self.do = nn.Dropout(self.dropout)
        self.ido = nn.Dropout(self.input_dropout)


    def forward(self, dna, protein, dna_mask, protein_mask, cross_attn_mask):
        # torch.tensor
        # dna = [batch, max_d, dna_dim]
        # protein = [batch, max_p, protein_dim]
        # dna_mask = [batch, 1, max_d, max_d]
        # protein_mask = [batch, 1, max_p, max_p]
        # cross_attn_mask = [batch, 1, max_p, max_d]

        if self.return_attention:
            enc_src, attention_dna = self.encoder1(dna, dna_mask)
            # enc_src = [batch size, dna len, hid_dim]

            enc_trg, attention_protein = self.encoder2(protein, protein_mask)
            # enc_trg = [batch size ,pro len, hid_dim]

            out, attention_cross = self.decoder(enc_trg, enc_src, protein_mask, cross_attn_mask)

            return torch.squeeze(out), attention_protein, attention_dna, attention_cross  
        else:
            # add random masking (optional)
            dna = self.ido(dna)
            protein = self.ido(protein)
            
            enc_src = self.encoder1(dna, dna_mask)
            enc_trg = self.encoder2(protein, protein_mask)
            out = self.decoder(enc_trg, enc_src, protein_mask, cross_attn_mask)
            return torch.squeeze(out)
    
        
def run_epoch(data_iter,model,optimizer,scheduler,use_cuda,mode="train",accum_iter=1):
    """Train a single epoch"""
    total_loss, n_accum, count = 0,0,0
    y_labels, y_preds = [], []
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_iter):

        dna_mask = d_mask.unsqueeze(1).unsqueeze(2) # dna_mask = [batch, 1, 1, max_d]
        protein_mask = p_mask.unsqueeze(1).unsqueeze(3)    # protein_mask = [batch, 1, max_p, 1]
        cross_attn_mask = torch.matmul(protein_mask, dna_mask)  # cross_attn_mask = [batch, 1, max_p, max_d]       
        dna_mask = torch.matmul(dna_mask.permute(0,1,3,2), dna_mask) # dna_mask = [batch, 1, max_d, max_d]
        protein_mask = torch.matmul(protein_mask, protein_mask.permute(0,1,3,2)) # protein_mask = [batch, 1, max_p, max_p]
        label = torch.from_numpy(np.array(label)).float() # label = [batch, num_labels]

        if use_cuda:
            d = d.cuda()
            p = p.cuda()
            cross_attn_mask = cross_attn_mask.cuda()
            dna_mask = dna_mask.cuda()
            protein_mask = protein_mask.cuda()
            label = label.cuda()  

        score = model(d, p, dna_mask, protein_mask, cross_attn_mask)
#         # manually set higher weight on binding, however found model learned nothing about family
#         class_weights=torch.ones(label.size(1))
#         class_weights[0] = 5

#         num_per_class = 1+torch.sum(label, axis=0) # add pseudo-number to avoid inf
#         total_num = torch.sum(num_per_class)
#         class_weights = total_num/num_per_class
#         class_weights[0] = class_weights[0]*5  # set higher weight on binding
#         class_weights = class_weights.cuda()
#         print(class_weights.shape)
#         loss_fct = torch.nn.BCEWithLogitsLoss(weight=class_weights)  
        loss_fct = torch.nn.BCEWithLogitsLoss()  
#         pos_weight = torch.tensor([60]).cuda()
#         loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)  
        loss = loss_fct(score, label) 

        if mode == "train" or mode == "train+log":
            loss.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
            scheduler.step()

        total_loss += loss
        count += 1
#         print(loss)
           
        if (i+1) % 1000 == 0 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
#             print(class_weights)
            print(
                (
                    "Epoch Step: %4d | Loss: %6.3f "
                    + "| Learning Rate: %6.1e"
                )
                % (i+1, loss, lr)
            )
#             print('predicted interaction:',logits.to('cpu').data.numpy())
#             print('predicted interaction:',label.to('cpu').data.numpy())
#             print('dna sequence:',d.to('cpu').data.numpy())
#             print('protein sequence:',p.to('cpu').data.numpy())


        if mode == "test":
            y_labels.extend(label.to('cpu').data.numpy())
            y_preds.extend(torch.sigmoid(score).to('cpu').data.numpy())


        
    if mode == "test":
        y_pred = np.array(y_preds) # [num_examples, num_class]
        y_label = np.array(y_labels)
        # print(y_label[:10])
        # print(y_label.shape)
        # print(y_pred[:10])
        # print(y_pred.shape)        
        accuracy, sensitivity, specificity, ROC, PRC, MCC, F1 = evaluate(y_pred, y_label)
        return accuracy, sensitivity, specificity, ROC, PRC, MCC, F1, total_loss/count       

    return total_loss / count
          
def predict(data_iter,model,use_cuda,protein_graph_node_ft=None,verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        y_labels, y_preds = [], []
        
        for i, (d, p, d_mask, p_mask, label) in enumerate(data_iter):

            dna_mask = d_mask.unsqueeze(1).unsqueeze(2) # dna_mask = [batch, 1, 1, max_d]
            protein_mask = p_mask.unsqueeze(1).unsqueeze(3)    # protein_mask = [batch, 1, max_p, 1]
            cross_attn_mask = torch.matmul(protein_mask, dna_mask)  # cross_attn_mask = [batch, 1, max_p, max_d]       
            dna_mask = torch.matmul(dna_mask.permute(0,1,3,2), dna_mask) # dna_mask = [batch, 1, max_d, max_d]
            protein_mask = torch.matmul(protein_mask, protein_mask.permute(0,1,3,2)) # protein_mask = [batch, 1, max_p, max_p]
            label = torch.from_numpy(np.array(label)).float()

            if use_cuda:
                d = d.cuda()
                p = p.cuda()
                cross_attn_mask = cross_attn_mask.cuda()
                dna_mask = dna_mask.cuda()
                protein_mask = protein_mask.cuda()
                label = label.cuda()  

            score = model(d, p, dna_mask, protein_mask, cross_attn_mask)
            y_labels.extend(label.to('cpu').data.numpy())
            y_preds.extend(torch.sigmoid(score).to('cpu').data.numpy())

        y_preds = np.array(y_preds) # [num_examples, num_class]
        y_labels = np.array(y_labels)
        
        output = pd.DataFrame({'pred_label':y_preds, 'true_label': y_labels})
        return output

    
    
def predict_with_attention(data_iter,model,use_cuda,protein_graph_node_ft=None,verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        y_labels, y_preds, att_pro, att_dna, att_cross, p_start, dna_seqs, protein_names = [], [], [], [], [], [], [], []
 
        for i, (d, p, d_mask, p_mask, label, p_start_idx, dna_seq, protein_name) in enumerate(data_iter):
        
            dna_seqs.extend(dna_seq)
            protein_names.extend(protein_name)
            
            dna_mask = d_mask.unsqueeze(1).unsqueeze(2) # dna_mask = [batch, 1, 1, max_d]
            protein_mask = p_mask.unsqueeze(1).unsqueeze(3)    # protein_mask = [batch, 1, max_p, 1]
            cross_attn_mask = torch.matmul(protein_mask, dna_mask)  # cross_attn_mask = [batch, 1, max_p, max_d]       
            dna_mask = torch.matmul(dna_mask.permute(0,1,3,2), dna_mask) # dna_mask = [batch, 1, max_d, max_d]
            protein_mask = torch.matmul(protein_mask, protein_mask.permute(0,1,3,2)) # protein_mask = [batch, 1, max_p, max_p]
            label = torch.from_numpy(np.array(label)).float()

            if use_cuda:
                d = d.cuda()
                p = p.cuda()
                cross_attn_mask = cross_attn_mask.cuda()
                dna_mask = dna_mask.cuda()
                protein_mask = protein_mask.cuda()
                label = label.cuda()  

            score, attention_protein, attention_dna, attention_cross = model(d, p, dna_mask, protein_mask, cross_attn_mask)
            y_labels.extend(label.to('cpu').data.numpy())
            y_preds.extend(torch.sigmoid(score).to('cpu').data.numpy())
            att_pro.extend(attention_protein.to('cpu').data.numpy())   
            att_dna.extend(attention_dna.to('cpu').data.numpy())   
            att_cross.extend(attention_cross.to('cpu').data.numpy())   
            p_start.extend(p_start_idx.to('cpu').data.numpy())
            
        output = pd.DataFrame({'dna':dna_seqs, 'protein': protein_names, 'pred_label':y_preds, 'true_label': y_labels, 'label': [1 if i>0.5 else 0 for i in y_preds], 'p_start_idx': p_start, 'attention_dna': att_dna, 'attention_protein': att_pro, 'attention_cross': att_cross})

#         y_preds = np.array(y_preds) # [num_examples, num_class]
#         y_labels = np.array(y_labels)

        # print(output)
        return output
        

        
######################### end of models #############################
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
    start_pos = 0
    seq_length = x.shape[0]
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
    return (x, np.asarray(input_mask), start_pos) if return_start_index else (x, np.asarray(input_mask), )

    
class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti, max_d, max_p, dna_embs, pro_embs, df_family, return_attention):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.max_d = max_d
        self.max_p = max_p
        self.dna_embs = dna_embs
        self.pro_embs = pro_embs
        self.df_family = df_family
        self.return_start_index = return_attention
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        dna_seq = self.df.iloc[index]['dna']
        d = self.dna_embs.loc[self.dna_embs['dna']==dna_seq,'dna_embedding'].iloc[0]
        d_v, input_mask_d = unify_dim_embedding(d,self.max_d)
        
        protein = self.df.iloc[index]['protein']
        protein_idx = self.pro_embs.index[self.pro_embs['protein']==protein].tolist()[0]
        p = torch.from_numpy(self.pro_embs.loc[protein_idx,'protein_embedding'])
#         p_v, input_mask_p = unify_dim_embedding(p,self.max_p, self.return_start_index)
        p_outs = unify_dim_embedding(p,self.max_p, self.return_start_index)
        p_v, input_mask_p = p_outs[0], p_outs[1] # p_embedding, mask, p_start_idx
        y = self.labels[index]
#         y = np.append(np.array(self.labels[index]),np.array(self.df_family.loc[protein]))
        
        return (d_v, p_v, input_mask_d, input_mask_p, y, p_outs[2], dna_seq, protein) if self.return_start_index else (d_v, p_v, input_mask_d, input_mask_p, y, )

def create_datasets(batch_size, max_d, max_p, df, dna_embs, pro_embs, df_family, return_attention):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0, 
          'drop_last': True,
          'pin_memory': True}
    
    dna_seqs = dna_embs.dna.unique().tolist()
    df = df.loc[df.dna.isin(dna_seqs)]
            
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p, dna_embs, pro_embs, df_family, return_attention)
    generator = data.DataLoader(dataset, **params)
    
    return generator, len(dataset)


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )



def load_dpi_data(file,pros):
    dpi = pd.read_pickle(file)
    dpi = dpi.loc[dpi['protein'].isin(pros)]

    return dpi

        
        
if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    """ create config"""
    config = get_config()
    print(config)

    device = config['device']
    batch = config['batch_size']
    lr = config['lr']
    max_d = config['max_dna_seq']
    max_p = config['max_protein_seq']
    iteration_per_split = config['iteration_per_split']
    files_per_split = config['files_per_split']
    
    """ create model, optimizer and scheduler"""
    model = init_model(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, config['hid_dim'], factor=1, warmup=config["warmup"]))    
    
    """Output files."""
    mid = tag+'-lr='+str(lr)+',epoch='+str(config['iteration_per_split'])+',dropout='+str(config['dropout'])+':'+str(config['input_dropout'])+',hid_dim='+str(config['hid_dim'])+',n_layer='+str(config['n_layers'])+',n_heads='+str(config['n_heads'])+',batch='+str(batch)+',input='+input_data+',max_dna='+str(max_d)+',max_protein='+str(max_p)
    
    file_AUCs = 'output/result/AUCs--'+mid+'.txt'
    file_model = 'output/model/'+mid+'.pt'
    
    
    AUC = ('total Epoch\tCurrent iteration\tTime(sec)\tLoss_train\tLoss_val\taccuracy\tsensitivity\tspecificity\tROC\tPRC\tMCC\tF1')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

#     """load trained model"""
#     model.load_state_dict(torch.load(file_model))
# # #     model.load_state_dict(torch.load(prev_model))
#     model.to(device)
        
    """Start training."""
    print('Training...')
    start = timeit.default_timer()
#     max_AUC_dev = 0
    min_valid_loss = 100
    num_split = math.ceil(len(files_dna_train)/files_per_split)
    
    """load protein embedding database"""
#     pro_embs = pd.read_pickle(maindir +input_data+'/'+input_data + '_protein_embedding_mean.pkl').reset_index()
    pro_embs = pd.read_pickle('/new-stg/home/cong/DPI/dataset/Encode3_Encode4_ChIP690_protein_embedding_mean.pkl').reset_index()

    # create initial graph embedding: (node num, 384)
    protein_graph_node_ft = torch.from_numpy(np.stack(pro_embs.apply(lambda row: np.mean(row['protein_embedding'], axis=0) , axis = 1)))
#     pros = pro_embs.protein.unique().tolist()  # the scope of all proteins

        
#     with open("/new-stg/home/cong/DPI/dataset/ChIP_690/random_2_frac_07_119_proteins.txt") as file:
    with open(train_pro_file) as file:        
        pros_train = [line.rstrip() for line in file]  
#     with open(valid_pro_file) as file:
#         pros_valid = [line.rstrip() for line in file] 

    """load TF family labels"""
#     df_family = pd.read_pickle('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/data/TF_family_label.pkl')
#     df_family = pd.read_pickle('/new-stg/home/cong/DPI/dataset/all_TF_family_label.pkl')
    df_family = pd.read_pickle('/new-stg/home/cong/DPI/dataset/all_TF_top_family_label.pkl') # reduce the total num of families

    
    """load interaction data"""
#     dpi_train = load_dpi_data(maindir+'../data/'+input_data+'.pkl', pros) # original class balanced data by dna
#     dpi_valid = load_dpi_data(maindir+'../data/'+valid_data+'.pkl', pros)

    dpi_train = load_dpi_data(maindir+'../data/'+input_data+'.pkl', pros_train)
    
#     dpi_valid = load_dpi_data(maindir+'../data/'+valid_data+'.pkl', pros_valid)
    dpi_valid = load_dpi_data(maindir+'../data/'+valid_data+'.pkl', pros_train)
#     dpi_valid = load_dpi_data(maindir+'../data/'+valid_data+'.pkl', pros_train+pros_valid) # use both proteins from train and valid

#     dpi_train = load_dpi_data(maindir+'../data/all_'+input_data+'_long.pkl', pros) # all the data, same as deepsea
#     dpi_valid = load_dpi_data(maindir+'../data/all_'+valid_data+'_long.pkl', pros) 

#     dpi_train = load_dpi_data(maindir, input_data+'_full_balance_by_protein', pros) # ratio balance data
#     dpi_valid = load_dpi_data(maindir, valid_data+'_full_balance_by_protein', pros) # ratio balance data
    
    print('In total: '+str(dpi_train.protein.unique().size)+' proteins; '+str(dpi_train.dna.unique().size)+' dna sequences; '+str(len(dpi_train))+' examples are included in training.')
    print('In total: '+str(dpi_valid.protein.unique().size)+' proteins; '+str(dpi_valid.dna.unique().size)+' dna sequences; '+str(len(dpi_valid))+' examples are included in validation.')

    
    # check gpu memory usage
    print(f"GPU memory usage percent: {100*torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}%", flush=True)
    j = 0
            
    for i in range(num_split):     
#         chosen_files = files_dna_train[i*files_per_split: min((i+1)*files_per_split, len(files_dna_train))]
        chosen_files = random.sample(files_dna_train, files_per_split) # random sampling
        li = []
        for filename in chosen_files:
            print(filename)
            df = pd.read_pickle(filename)
            li.append(df)
        dna_embs_train = pd.concat(li, axis=0, ignore_index=True)        
        dataset_train, train_size = create_datasets(batch, max_d, max_p, dpi_train, dna_embs_train, pro_embs, df_family, config['return_attention'])

        # each valid file has too many examples, caused out-of-ram issue. will use only partial
        if i % 2 == 0:
            dna_embs_valid = pd.read_pickle(maindir+valid_data+'/'+valid_data + '_dna_embedding_6mer_'+str(j % len(files_dna_valid)+1)+'.pkl')
            dna_embs_valid_partial = dna_embs_valid.sample(n=config['valid_dna_size'], random_state=123)
            del dna_embs_valid
            dataset_valid, valid_size = create_datasets(batch, max_d, max_p, dpi_valid, dna_embs_valid_partial, pro_embs, df_family, config['return_attention'])

            print(f"Loaded valid dataset split {j+1}", flush=True)
            j = j + 1
                    
        print(f"GPU memory usage percent: {100*torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}%", flush=True)

        for epoch in range(1, iteration_per_split + 1):
            model.train()
            print(f"Split {i+1} Epoch {epoch} Training ====", flush=True)
            print(f"training size: {train_size}", flush=True)
            print(f"valid size: {valid_size}", flush=True)

            loss_train = run_epoch(
                dataset_train,
                model,
                optimizer,
                scheduler,
                use_cuda,
                mode="train",
                accum_iter=1
            )
            print(f"GPU memory usage percent: {100*torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}%", flush=True)
            GPUtil.showUtilization()
            torch.cuda.empty_cache()


            print(f"Split {i+1} Epoch {epoch} Validation ====", flush=True)
            with torch.set_grad_enabled(False):
                model.eval()
                accuracy, sensitivity, specificity, ROC, PRC, MCC, F1, loss_dev = run_epoch(
                    dataset_valid,
                    model,
                    optimizer,
                    scheduler,
                    use_cuda,
                    mode="test",
                    accum_iter=1
                )
            end = timeit.default_timer()
            time = end - start

            loss_train = loss_train.to('cpu').data.numpy()
            loss_dev = loss_dev.to('cpu').data.numpy()
    #         train_loss.append(loss_train)
    #         val_loss.append(loss_dev)
    #         val_auc.append(AUC_dev)
    #         val_accuracy.append(accuracy)
            print('train loss:', loss_train)
            print('val loss:', loss_dev)
            AUCs = [i*iteration_per_split+epoch, epoch, time, loss_train, loss_dev, accuracy, sensitivity, specificity, ROC, PRC, MCC, F1]

            with open(file_AUCs, 'a') as f:
                f.write('\t'.join(map(str, AUCs)) + '\n')

            if loss_dev < min_valid_loss:
                torch.save(model.state_dict(), file_model)
                min_valid_loss = loss_dev
            print('\t'.join(map(str, AUCs)))
        
    print('Finished training...')    
    
    # print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print(model)

    # save the last model
    torch.save(model.state_dict(), 'output/model/'+mid+'_final.pt')
    
    # plot the learning curve
    sns.set(font_scale=1.5)
    sns.set_style(style='ticks')
    get_learning_curve(file_AUCs)




