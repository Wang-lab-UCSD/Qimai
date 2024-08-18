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


# from models import *
import pickle
import matplotlib
import matplotlib.pyplot as plt

dataset = 'ChIP_690'
# input_data = 'train'
input_data = 'valid'
tag = 'main_v0_deepsea_'+dataset
files = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+input_data+'*pkl', recursive=True)

files.sort()
num = len(files)
print('-------'+dataset+'-------')
print('training files: '+str(num))
print(files)

valid_data = 'train'
# valid_data = 'valid'
# files_valid = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+valid_data+'*pkl', recursive=True)
files_valid = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+valid_data+'*_[0-9].pkl', recursive=True)

print('validation files: '+str(len(files_valid)))
print(files_valid)


def config():
    config = {}
    config['batch_size'] = 16
    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
    
#     config['max_protein_seq'] = 512
    config['max_protein_seq'] = 768
    
    config["warmup"]  = 10000
    config['iteration_per_split'] = 5
    config['files_per_split'] = 5
    config['files_per_split_valid'] = 3

    config['hid_dim'] = 240
    config['dropout'] = 0.1
    config['lr'] = 1e-2
    config['pf_dim'] = 2048
    config['n_layers'] = 6
    config['n_heads'] = 12
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return config



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
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

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
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

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
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.ln1 = LayerNorm(hid_dim)
        self.ln2 = LayerNorm(hid_dim)
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)

        
    def forward(self, trg, trg_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, 1, dna sent len, dna sent len]
        # src_mask = [batch size, 1, protein len, protein len]

        trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln2(trg + self.do2(self.pf(trg)))

        return trg

class Encoder(nn.Module):
    def __init__(self, dna_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)      
        self.ft = nn.Linear(dna_dim, hid_dim)
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device))

        
    def forward(self, trg, trg_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        
        trg = self.ft(trg)
        # trg = [batch size, dna len, hid dim]

        for layer in self.layer:
            trg = layer(trg, trg_mask)
            
        return trg
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.ln1 = LayerNorm(hid_dim)
        self.ln2 = LayerNorm(hid_dim)
        self.ln3 = LayerNorm(hid_dim)
        
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.do3 = nn.Dropout(dropout)
        
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        

    def forward(self, trg, src, trg_mask=None, cross_attn_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        # src = [batch_size, protein len, hid_dim] 
        # trg_mask = [batch size, 1, dna sent len, dna sent len]
        # cross_attn_mask = [batch, 1, dna sent len, protein len]

        trg = self.ln1(trg + self.do1(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln2(trg + self.do2(self.ea(trg, src, src, cross_attn_mask)))
        trg = self.ln3(trg + self.do3(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ dna feature extraction."""
    def __init__(self, dna_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.ln = LayerNorm(hid_dim)
        self.output_dim = dna_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.layer = nn.ModuleList()
        for _ in range(n_layers):
            self.layer.append(DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device))

        # self.ft = nn.Linear(dna_dim, hid_dim)
        self.ft = nn.Linear(hid_dim, hid_dim)
        self.output = nn.Sequential(
            nn.Linear(self.hid_dim, 256),
            nn.ReLU(True),
#             nn.Dropout(p=self.dropout),
            
            nn.Linear(256, 64),
            nn.ReLU(True),
#             nn.Dropout(p=self.dropout),
            
            nn.Linear(64, 32),
            nn.ReLU(True),
#             nn.Dropout(p=self.dropout),
            
            #output layer
            nn.Linear(32, 1)
        )
        

    def forward(self, trg, src, trg_mask=None,cross_attn_mask=None):
        # trg = [batch_size, dna len, dna_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, 1, dna sent len, dna sent len]
        # cross_attn_mask = [batch, 1, dna sent len, protein len]        
        trg = self.ft(trg)

        # trg = [batch size, dna len, hid dim]

        for layer in self.layer:
            trg = layer(trg, src, trg_mask, cross_attn_mask)

        # trg = [batch size, dna len, hid dim]

#         label, _ = torch.max(trg, dim=1)
        trg_mask_2d = trg_mask[:,0,:,0]
        label = torch.sum(trg*trg_mask_2d[:,:,None], dim=1)/trg_mask_2d.sum(dim=1, keepdims=True)
        label = label.unsqueeze(1)
        label = self.output(label)
        label = torch.squeeze(label)
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
        self.batch_size = config['batch_size']
        self.device = config['device']
        
        self.encoder1 = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device)
        self.encoder2 = Encoder(self.dna_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device)
        
        self.decoder = Decoder(self.dna_dim, self.hid_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout, self.device)



    def forward(self, dna, protein, dna_mask, protein_mask, cross_attn_mask):
        # torch.tensor
        # dna = [batch, max_d, dna_dim]
        # protein = [batch, max_p, protein_dim]
        # dna_mask = [batch, 1, max_d, max_d]
        # protein_mask = [batch, 1, max_p, max_p]
        # cross_attn_mask = [batch, 1, max_d, max_p]

#         with autocast():
        enc_src = self.encoder1(protein, protein_mask)
        # enc_src = [batch size, protein len, hid_dim]

        enc_trg = self.encoder2(dna, dna_mask)
        # enc_trg = [batch size ,dna_num, hid_dim]

        out = self.decoder(enc_trg, enc_src, dna_mask, cross_attn_mask)

        return out
           
        
def run_epoch(data_iter,model,optimizer,scheduler,use_cuda,mode="train",accum_iter=1):
    """Train a single epoch"""
#     start = time.time()
    total_loss, n_accum, count = 0,0,0
    y_label, y_pred = [], []
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_iter):
        dna_mask = d_mask.unsqueeze(1).unsqueeze(3) # dna_mask = [batch, 1, max_d, 1]
        protein_mask = p_mask.unsqueeze(1).unsqueeze(2)    # protein_mask = [batch, 1, 1, max_p]
        cross_attn_mask = torch.matmul(dna_mask, protein_mask)  # cross_attn_mask = [batch, 1, max_d, max_p]        
        protein_mask = torch.matmul(protein_mask.permute(0,1,3,2), protein_mask) # protein_mask = [batch, 1, max_p, max_p]
        dna_mask = torch.matmul(dna_mask, dna_mask.permute(0,1,3,2)) # dna_mask = [batch, 1, max_d, max_d]
        label = Variable(torch.from_numpy(np.array(label)).float())

        if use_cuda:
            d = d.cuda()
            p = p.cuda()
            cross_attn_mask = cross_attn_mask.cuda()
            dna_mask = dna_mask.cuda()
            protein_mask = protein_mask.cuda()
            label = label.cuda()  
            
        optimizer.zero_grad(set_to_none=True)
        
        # forward pass and loss calculation
#         with autocast():
        score = model(d, p, dna_mask, protein_mask, cross_attn_mask)
        m = torch.nn.Sigmoid()
        logits = m(score)
        loss = F.binary_cross_entropy_with_logits(score, label)   
        
        LOSS = loss.to('cpu').data.numpy()
        total_loss += LOSS
        count += 1
#         print(loss)

#         # backward pass and optimization
#         if mode == "train" or mode == "train+log":
#             scaler.scale(loss).backward()
#             if i % accum_iter == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 n_accum += 1
#             scheduler.step()

        if mode == "train" or mode == "train+log":
            loss.backward()
            if i % accum_iter == 0:
                optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
                n_accum += 1
            scheduler.step()


        
        if mode == "test":
            y_label.extend(label.to('cpu').data.numpy())
            y_pred.extend(logits.to('cpu').data.numpy())
            
        # print message
        if (i+1) % 100 == 0 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
#             elapsed = time.time() - start
#             print(
#                 (
#                     "Epoch Step: %4d | Loss: %6.3f "
#                     + "| Sec: %6.1f | Learning Rate: %6.1e"
#                 )
#                 % (i+1, LOSS, elapsed, lr)
#             )
            print(
                (
                    "Epoch Step: %4d | Loss: %6.3f "
                    + "| Learning Rate: %6.1e"
                )
                % (i+1, LOSS, lr)
            )
#             print('predicted interaction:',logits.to('cpu').data.numpy())
#             print('predicted interaction:',label.to('cpu').data.numpy())
#             print('dna sequence:',d.to('cpu').data.numpy())
#             print('protein sequence:',p.to('cpu').data.numpy())
#             start = time.time()
        
        
    if mode == "test":
        try:
            ROC = roc_auc_score(y_label, y_pred)
            PRC = average_precision_score(y_label, y_pred)

            fpr, tpr, thresholds = roc_curve(y_label, y_pred)

            precision = tpr / (tpr + fpr)

            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

            # get the the threshold
            J = tpr - fpr
            ix = np.argmax(J)
            # thred_optim = thresholds[5:][np.argmax(f1[5:])]
            # thred_optim = 0.5
            thred_optim = thresholds[ix]

            # y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            y_pred_s = [1 if i>thred_optim else 0 for i in y_pred]

            auc_k = metrics.auc(fpr, tpr)
            cm1 = confusion_matrix(y_label, y_pred_s)
            total1=sum(sum(cm1))
            #####from confusion matrix calculate accuracy
            accuracy1=(cm1[0,0]+cm1[1,1])/total1
            sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

            F1 = f1_score(y_label, y_pred_s)
            recall = recall_score(y_label, y_pred_s)        
            MCC = matthews_corrcoef(y_label, y_pred_s)

            return accuracy1, ROC, PRC, MCC, recall, F1, total_loss / count       
        
        except ValueError:
            return 0, 0, 0, 0, 0, 0, total_loss/count
        
    return total_loss / count
  

def init_model(config):

    model = Predictor(**config)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

    
# load DNABERT and alphaFold embeddings
def unify_dim_embedding(x,crop_size):
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
    return x, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti, max_d, max_p):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.max_d = max_d
        self.max_p = max_p
        
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
        p_v, input_mask_p = unify_dim_embedding(p,self.max_p)
        y = self.labels[index]
        
#         print(d_v.shape)
#         print(p_v.shape)
#         print(input_mask_d.shape)
#         print(input_mask_p.shape)
#         print(y.shape)
        return d_v, p_v, input_mask_d, input_mask_p, y


def create_datasets(batch_size, df):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0, 
          'drop_last': True,
          'pin_memory': True}
        
#     print("dataset size: ", df.shape[0])
    
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p)
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


def load_embedding_df(files,i,files_per_split):
    chosen_files = files[i*files_per_split: min((i+1)*files_per_split, len(files))]
    li = []
    for filename in chosen_files:
        print(filename)
        df = pd.read_pickle(filename)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    filter = frame['dna'].str.contains('N')
    frame = frame[~filter]
    dataset_train, dataset_size = create_datasets(batch, frame)
    return dataset_train, dataset_size


        
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
    
    protein_dim = config['protein_dim']
    dna_dim = config['dna_dim']
    hid_dim = config['hid_dim']
    device = config['device']
    batch = config['batch_size']
    lr = config['lr']
    dropout = config['dropout']
    n_layers = config['n_layers']
    max_d = config['max_dna_seq']
    max_p = config['max_protein_seq']
    iteration_per_split = config['iteration_per_split'] 
    files_per_split = config['files_per_split'] 
    files_per_split_valid = config['files_per_split_valid'] 
    

    """ create model, optimizer and scheduler"""
    model = init_model(config)
    model.to(device)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, config['hid_dim'], factor=1, warmup=config["warmup"]))    
    
    """Output files."""
    mid = tag+'-lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)
    file_AUCs = 'output/result/AUCs--'+mid+'.txt'
    file_model = 'output/model/'+mid+'.pt'
    
    AUC = ('total Epoch\tCurrent iteration\tTime(sec)\tLoss_train\tLoss_val\taccuracy_dev\tROC_dev\tPRC_dev\tMCC_dev\trecall_dev\tF1_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

        
    """Start training."""
    print('Training...')
    start = timeit.default_timer()
    
    # chosse MCC as the main metric
    max_MCC_dev = 0
#     train_loss, val_loss, val_auc, val_accuracy = [], [], [], []
    num_split = math.ceil(len(files)/files_per_split)
    valid_total_splits = math.ceil(len(files_valid)/files_per_split_valid)
    
    # evaluate on the same valid data for several epochs and change to another batch
    j = 0
    print(f"Loaded valid dataset split {j+1}", flush=True)
    dataset_valid, valid_size = load_embedding_df(files_valid,j,files_per_split_valid)

    for i in range(0, num_split):
        
        # for every 10 splits, change valid dataset
        if (i+1) % 10 == 0:
            print(f"Loaded valid dataset split {j+1}", flush=True)
            dataset_valid, valid_size = load_embedding_df(files_valid,(j+1) % valid_total_splits, files_per_split_valid)
            j = j + 1
      
        """Load preprocessed data."""
        print(f"Loaded train dataset split {i+1}", flush=True)
        dataset_train, train_size = load_embedding_df(files,i,files_per_split)

        for epoch in range(1, iteration_per_split + 1):
            model.train()
            print(f"Split {i+1} Epoch {epoch} Training ====", flush=True)

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
                accuracy, ROC, PRC, MCC, recall, f1, loss_dev = run_epoch(
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


            print('train loss:', loss_train)
            print('val loss:', loss_dev)
            AUCs = [i*iteration_per_split+epoch, epoch, time, loss_train, loss_dev, accuracy, ROC, PRC, MCC, recall, f1]
            with open(file_AUCs, 'a') as f:
                f.write('\t'.join(map(str, AUCs)) + '\n')

            if MCC > max_MCC_dev:
                torch.save(model.state_dict(), file_model)
                max_MCC_dev = MCC
            print('\t'.join(map(str, AUCs)))
        
    print('Finished training...')    
    
    # print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print(model)
