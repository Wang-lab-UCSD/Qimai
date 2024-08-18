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

tag = 'model_deepsea'
# tag = 'ChIP_690_full_model_deepsea'
# dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/data_with_embedding/one_percent_data/'
# input_data = 'val_v2_frac_01'  
# input_data = 'val_v2_pro_10'  
# input_data = 'val_v2_6mer'
# input_data = 'train_v2_6mer'
# eval_data = 'val_v2_6mer_1'

# dataFolder = '/new-stg/home/cong/DPI/dataset/Encode3/embeddings/'
# input_data = '100_per_exp_train_6mer'  
# eval_data = '100_per_exp_test_6mer'

# dataFolder = '/new-stg/home/cong/DPI/dataset/Encode3/embeddings_512bp/'
# input_data = '100_per_exp_max_512bp_train_6mer'  
# eval_data = '100_per_exp_max_512bp_test_6mer'
# input_data = '100_per_exp_1258exp_RPM_train_6mer'  
# eval_data = '100_per_exp_1258exp_RPM_test_6mer'
# num = 11


# files = glob.glob(dataFolder+input_data+'*pkl')

# combine ChIP-690 and Encode3 data together
# input_data = '100_per_exp_max_512bp_train_6mer'  
# eval_data = '100_per_exp_max_512bp_test_6mer'
# files = glob.glob('/new-stg/home/cong/DPI/dataset/**/'+input_data+'*pkl', recursive=True)

# dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/embeddings/'
# input_data = "MCF-7_train"
# input_data = "HepG2_valid_balanced"
# input_data = "MCF-7_valid_balanced"
# files = glob.glob(dataFolder+input_data+'*pkl', recursive=True)


# input_data = 'train'
input_data = 'train_min10'

dataset = 'ChIP_690'
# dataset = 'Encode3'
# input_data = 'train_s18'
f1 = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+input_data+'*pkl', recursive=True)
# f2 = glob.glob('/new-stg/home/cong/DPI/dataset/Encode4/deepsea/embeddings/'+input_data+'*pkl', recursive=True)
files=f1

files.sort()
num = len(files)
print('-------'+dataset+'-------')
print('training files: '+str(num))
print(files)

# valid_data = 'train_s1'
valid_data = 'valid'
files_valid = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+valid_data+'*pkl', recursive=True)
print('validation files: '+str(len(files_valid)))
print(files_valid)


# def config():
#     config = {}
#     config['batch_size'] = 16
#     config['dna_dim'] = 768
#     config['protein_dim'] = 384
#     config['max_dna_seq'] = 98
#     config['max_protein_seq'] = 512
#     config["warmup"]  = 5000
#     config['iteration_per_split'] = 10
#     config['files_per_split'] = 5
#     config['hid_dim'] = 192
#     config['dropout'] = 0.1
#     config['lr'] = 1e-2
#     config['pf_dim'] = 2048
#     config['n_layers'] = 3
#     config['n_heads'] = 12
#     config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     return config

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
    config['files_per_split'] = 5
    config['files_per_split_valid'] = 3
    
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

# split train and valid in each loading
# def create_datasets(batch_size, df):
#     print('--- Data Preparation ---')

#     params = {'batch_size': batch_size,
#               'shuffle': True,
#               'num_workers': 1, 
#               'drop_last': True,
#               'pin_memory': True}

    
# #     df = pd.read_pickle(dataFolder + path + '_6.pkl')
# #     # df = pd.read_csv(dataFolder + path + suffix)
# #     # df = pd.read_pickle(dataFolder + path + '_6_ft.pkl')
# #     filter = df['dna'].str.contains('N')
# #     df = df[~filter]
    
#     df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
#     print("training size: ", df_train.shape[0])
#     print("validation size: ", df_val.shape[0])
    
#     training_set = BIN_Data_Encoder(np.array([i for i in range(df_train.shape[0])]), df_train.label.values, df_train, max_d, max_p)
#     training_generator = data.DataLoader(training_set, **params)

#     validation_set = BIN_Data_Encoder(np.array([i for i in range(df_val.shape[0])]), df_val.label.values, df_val, max_d, max_p)
#     validation_generator = data.DataLoader(validation_set, **params)
    
#     return training_generator, validation_generator

def create_datasets(batch_size, df):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8, 
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

# def test_accuracy(model,config):
#     # load test data
#     params = {'batch_size': config["batch_size"],
#               'shuffle': True,
#               'num_workers': 1, 
#               'drop_last': True,
#               'pin_memory': True}
#     df = pd.read_pickle(dataFolder + eval_data + '.pkl')
#     # df = pd.read_csv(dataFolder + path + suffix)
#     # df = pd.read_pickle(dataFolder + path + '_6_ft.pkl')
#     filter = df['dna'].str.contains('N')
#     df = df[~filter]
        
    
#     dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, config["max_dna_seq"], config["max_protein_seq"])
#     test_data = data.DataLoader(dataset, **params)
    
#     # run model 
#     # optimizer and scheduler are not needed, just used for place holders
#     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9, 0.98), eps=1e-6)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
#             step, 192, factor=1, warmup=4000))    
#     use_cuda = torch.cuda.is_available() 
    
#     with torch.set_grad_enabled(False):
#         model.eval()
#         accuracy, AUC_dev, PRC_dev, MCC, recall, f1, loss_dev = run_epoch(
#             test_data,
#             model,
#             optimizer,
#             scheduler,
#             use_cuda,
#             mode="test",
#             accum_iter=1
#         )
#     print('val loss: ', loss_dev)
#     return accuracy

        
if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    use_cuda = torch.cuda.is_available()

#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#         print('The code uses GPU...')
#     else:
#         device = torch.device('cpu')
#         print('The code uses CPU!!!')
        
    """ create config"""
    config = config()
    print(config)

    device = config['device']
    batch = config['batch_size']
    lr = config['lr']
    max_d = config['max_dna_seq']
    max_p = config['max_protein_seq']
    
    iteration_per_split = config['iteration_per_split'] 
    files_per_split = config['files_per_split'] 
    files_per_split_valid = config['files_per_split_valid'] 

    """ create model, optimizer and scheduler"""
    model = init_model(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, config['hid_dim'], factor=1, warmup=config["warmup"]))    
    
    """Output files."""
    mid = tag+'-lr='+str(lr)+',dropout='+str(config['dropout'])+',hid_dim='+str(config['hid_dim'])+',n_layer='+str(config['n_layers'])+',n_heads='+str(config['n_heads'])+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)
    
    file_AUCs = 'output/result/AUCs--'+mid+'.txt'
    prev_model = 'output/model/model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=746,max_dna=512,max_protein=768.pt'
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
    num_split = math.ceil(len(files)/files_per_split)
    valid_total_splits = math.ceil(len(files_valid)/files_per_split_valid)
    

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
    
    j = 0
    j = 4
    dataset_valid, valid_size = load_embedding_df(files_valid,j,files_per_split_valid)
    
    # check gpu memory usage
    print(f"GPU memory usage percent: {100*torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}%", flush=True)

#     for i in range(0, num_split):
    for i in range(40, num_split):
        
        # for every 10 splits, change valid dataset
        if (i+1) % 10 == 0:
            dataset_valid, valid_size = load_embedding_df(files_valid,(j+1) % valid_total_splits, files_per_split_valid)
            j = j + 1
            print(f"Loaded valid dataset split {j+1}", flush=True)
            
        """Load preprocessed data."""
        dataset_train, train_size = load_embedding_df(files,i,files_per_split)
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

