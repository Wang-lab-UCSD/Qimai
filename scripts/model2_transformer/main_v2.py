import torch
from torch.utils import data
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import os
import glob
import time
import math
import GPUtil

from models import *
import timeit
import pickle
import matplotlib
import matplotlib.pyplot as plt



# input_data = 'train'
input_data = 'train_min100'

# dataset = 'ChIP_690'
# dataset = 'Encode3'
dataset = 'Encode4'
tag = 'main_v2_deepsea_'+dataset
maindir = '/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'
files_dna_train = glob.glob(maindir+input_data+'/'+input_data+'_dna*pkl', recursive=True)

files_dna_train.sort()
print('-------'+dataset+'-------')
print('training dna embedding files: '+str(len(files_dna_train)))
[print(i) for i in files_dna_train]

# valid_data = 'train_s1'
valid_data = 'valid'
files_dna_valid = glob.glob(maindir+valid_data+'/'+valid_data+'_dna*pkl', recursive=True)
files_dna_valid.sort()
print('validation dna embedding files: '+str(len(files_dna_valid)))
[print(i) for i in files_dna_valid]


def config():
    config = {}
    config['batch_size'] = 16
    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
    
#     config['max_protein_seq'] = 512
    config['max_protein_seq'] = 768
    
    config["warmup"]  = 10000
    config['iteration_per_split'] = 2
    config['hid_dim'] = 240
    config['dropout'] = 0.1
    config['lr'] = 1e-2
    config['pf_dim'] = 2048
    config['n_layers'] = 6
    config['n_heads'] = 6
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        input_mask = [1] * crop_size
    return x, np.asarray(input_mask)
    

class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti, max_d, max_p, dna_embs):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.max_d = max_d
        self.max_p = max_p
        self.dna_embs = dna_embs
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        d = self.df.iloc[index]['dna_id']
        d = self.dna_embs.loc[self.dna_embs['dna_id']==d,'dna_embedding'].iloc[0]
        d_v, input_mask_d = unify_dim_embedding(d,self.max_d)
        
        p = self.df.iloc[index]['protein']
        p = torch.from_numpy(pro_embs.loc[pro_embs['protein']==p,'protein_embedding'].iloc[0])

        p_v, input_mask_p = unify_dim_embedding(p,self.max_p)
        y = self.labels[index]
        
#         print(d_v.shape)
#         print(p_v.shape)
#         print(input_mask_d.shape)
#         print(input_mask_p.shape)
#         print(y.shape)
        return d_v, p_v, input_mask_d, input_mask_p, y


def create_datasets(batch_size, df, dna_embs):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8, 
          'drop_last': True,
          'pin_memory': True}
            
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p, dna_embs)
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


        
if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    use_cuda = torch.cuda.is_available()

    """ create config"""
    config = config()
    print(config)

    device = config['device']
    batch = config['batch_size']
    lr = config['lr']
    max_d = config['max_dna_seq']
    max_p = config['max_protein_seq']
    iteration_per_split = config['iteration_per_split']
    
    """ create model, optimizer and scheduler"""
    model = init_model(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, config['hid_dim'], factor=1, warmup=config["warmup"]))    
    
    """Output files."""
    mid = tag+'-lr='+str(lr)+',dropout='+str(config['dropout'])+',hid_dim='+str(config['hid_dim'])+',n_layer='+str(config['n_layers'])+',n_heads='+str(config['n_heads'])+',batch='+str(batch)+',input='+input_data+',max_dna='+str(max_d)+',max_protein='+str(max_p)
    
    file_AUCs = 'output/result/AUCs--'+mid+'.txt'
    file_model = 'output/model/'+mid+'.pt'
    
    
    AUC = ('total Epoch\tCurrent iteration\tTime(sec)\tLoss_train\tLoss_val\taccuracy_dev\tAUC_dev\tPRC_dev\tMCC_dev\trecall_dev\tF1_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """load trained model"""
    model.load_state_dict(torch.load(file_model))
#     model.load_state_dict(torch.load(prev_model))
    model.to(device)
        
    """Start training."""
    print('Training...')
    start = timeit.default_timer()
    max_AUC_dev = 0
#     train_loss, val_loss, val_auc, val_accuracy = [], [], [], []
    
    """load protein embedding database"""
    pro_embs = pd.read_pickle(maindir +input_data+'/'+input_data + '_protein_embedding_mean.pkl')
    pros = pro_embs.protein.unique().tolist()  # the scope of all proteins
    
    """load interaction data"""
    def load_dpi_data(tag):
        dpi = pd.read_pickle(maindir+'../data/'+tag+'.pkl')
        dpi['dna_id'] = dpi.groupby('dna',sort=False).ngroup()+1
        dpi = dpi.loc[dpi['protein'].isin(pros)]
        
        return dpi
    
    dpi_train = load_dpi_data(input_data)
    dpi_valid = load_dpi_data(valid_data)
    print('In total: '+str(dpi_train.protein.unique().size)+' proteins; '+str(dpi_train.dna_id.unique().size)+' dna sequences; '+str(len(dpi_train))+' examples are included in training.')
    print('In total: '+str(dpi_valid.protein.unique().size)+' proteins; '+str(dpi_valid.dna_id.unique().size)+' dna sequences; '+str(len(dpi_valid))+' examples are included in validation.')

     
    def load_embedding_df(dna_embs,data):
        ids = dna_embs.dna_id.unique().tolist()
        df = data.loc[data.dna_id.isin(ids)]

#         # filter out the sequences with uncertainty, Encode4 didn't have this issue
#         filter = df['dna'].str.contains('N')
#         df = df[~filter]
        
        dataset_train, dataset_size = create_datasets(batch, df, dna_embs)
        return dataset_train, dataset_size

    
    # check gpu memory usage
    print(f"GPU memory usage percent: {100*torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}%", flush=True)
#     j = 0
    j = 2
#     for i in range(len(files_dna_train)):
    for i in range(6,len(files_dna_train)):
        
        dna_embs = pd.read_pickle(maindir+input_data+'/'+input_data + '_dna_embedding_6mer_'+str(i+1)+'.pkl')
        dataset_train, train_size = load_embedding_df(dna_embs, dpi_train)

        # for every 3 splits, change valid dataset
        if i % 3 == 0:
            dna_embs_valid = pd.read_pickle(maindir+valid_data+'/'+valid_data + '_dna_embedding_6mer_'+str((j+1) % len(files_dna_valid))+'.pkl')
            dataset_valid, valid_size = load_embedding_df(dna_embs_valid, dpi_valid)
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
                accuracy, AUC_dev, PRC_dev, MCC, recall, f1, loss_dev = run_epoch(
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
            AUCs = [i*iteration_per_split+epoch, epoch, time, loss_train, loss_dev, accuracy, AUC_dev, PRC_dev, MCC, recall, f1]
            with open(file_AUCs, 'a') as f:
                f.write('\t'.join(map(str, AUCs)) + '\n')

            if AUC_dev > max_AUC_dev:
                torch.save(model.state_dict(), file_model)
                max_AUC_dev = AUC_dev
            print('\t'.join(map(str, AUCs)))
        
    print('Finished training...')    
    
    # print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print(model)

    # save the last model
    torch.save(model.state_dict(), 'output/model/'+mid+'_final.pt')

