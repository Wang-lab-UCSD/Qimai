import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import numpy as np
import time
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# from Radam import *
# from lookahead import Lookahead


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

        
        enc_src = self.encoder1(protein, protein_mask)
        # enc_src = [batch size, protein len, hid_dim]
                
        enc_trg = self.encoder2(dna, dna_mask)
        # enc_trg = [batch size ,dna_num, hid_dim]

        out = self.decoder(enc_trg, enc_src, dna_mask, cross_attn_mask)

        return out


        
def run_epoch(data_iter,model,optimizer,scheduler,use_cuda,mode="train",accum_iter=1):
    """Train a single epoch"""
    start = time.time()
    total_loss, n_accum, count = 0,0,0
    y_label, y_pred = [], []
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_iter):
#         print(i)
#         print(d.shape)
#         print(p.shape)
#         print(d_mask.shape)
#         print(p_mask.shape)
#         print(label.shape)
        dna_mask = d_mask.unsqueeze(1).unsqueeze(3) # dna_mask = [batch, 1, max_d, 1]
        protein_mask = p_mask.unsqueeze(1).unsqueeze(2)    # protein_mask = [batch, 1, 1, max_p]
#         print(dna_mask.shape)
#         print(protein_mask.shape)
        cross_attn_mask = torch.matmul(dna_mask, protein_mask)  # cross_attn_mask = [batch, 1, max_d, max_p]
#         print(cross_attn_mask.shape)
        
        protein_mask = torch.matmul(protein_mask.permute(0,1,3,2), protein_mask) # protein_mask = [batch, 1, max_p, max_p]
        dna_mask = torch.matmul(dna_mask, dna_mask.permute(0,1,3,2)) # dna_mask = [batch, 1, max_d, max_d]
        
        label = Variable(torch.from_numpy(np.array(label)).float())
#         print(label.shape)

        if use_cuda:
            d = d.cuda()
            p = p.cuda()
            cross_attn_mask = cross_attn_mask.cuda()
            dna_mask = dna_mask.cuda()
            protein_mask = protein_mask.cuda()
            label = label.cuda()  
            
        score = model(d, p, dna_mask, protein_mask, cross_attn_mask)
#         m = torch.nn.Sigmoid()
#         logits = torch.squeeze(m(score))
        logits = torch.squeeze(score)
        
        loss_fct = torch.nn.MSELoss()   
        loss = loss_fct(logits, label) 

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
        
        if mode == "test":
            y_label.extend(label.to('cpu').data.numpy())
            y_pred.extend(logits.to('cpu').data.numpy())
            
        if (i+1) % 100 == 0 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %4d | Loss: %6.3f "
                    + "| Sec: %6.1f | Learning Rate: %6.1e"
                )
                % (i+1, loss, elapsed, lr)
            )
#             print('predicted interaction:',logits.to('cpu').data.numpy())
#             print('predicted interaction:',label.to('cpu').data.numpy())
#             print('dna sequence:',d.to('cpu').data.numpy())
#             print('protein sequence:',p.to('cpu').data.numpy())
            
            
            start = time.time()
        
        
    if mode == "test":
        r2 = r2_score(y_label,y_pred)
#         adj_r2_score = 1 - ((1-r2)*(n-1)/(n-k-1))
        MSE = mean_squared_error(y_label, y_pred)
        RMSE = mean_squared_error(y_label, y_pred, squared = False)
        MAE = mean_absolute_error(y_label, y_pred)
            
        return r2, MSE, RMSE, MAE, total_loss / count, y_label, y_pred       
        
    return total_loss / count
  
                
    