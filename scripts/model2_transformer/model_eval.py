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
from pathlib import Path
import GPUtil
import timeit
import pickle

from models import *
# from models_CR import *




####### set up path ###################
# TF = "FOS"
# TF = ["FOS","CTCF","RUNX3","AP2A","RAD21"]


# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer_1_segmentation_100fold,num_files=10,max_dna=512,max_protein=50.pt"
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer_1_segmentation_10fold,num_files=10,max_dna=512,max_protein=50.pt"
# model_path = "v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer_1,num_files=1,max_dna=512,max_protein=50.pt"
# model_path = 'v6-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_max_512bp_train_6mer,num_files=18,max_dna=512,max_protein=768.pt'
# model_path = 'model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=859,max_dna=512,max_protein=768.pt'

model_path = 'model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train,num_files=746,max_dna=512,max_protein=768.pt'
# model_path = 'ChIP_690_full_model_deepsea-lr=0.01,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=train_s18,num_files=40,max_dna=512,max_protein=768.pt'

# list all evaluation data
# eval_data = '100_per_exp_max_512bp_train_6mer'
# eval_data = '100_per_exp_max_512bp_train_6mer_1_segmentation_10fold_1'
# eval_data = '100_per_exp_max_512bp_train_6mer_1_segmentation_100fold_1'
# eval_data = "MCF-7_valid_6mer"
# eval_data = '100_per_exp_max_512bp_test_6mer'
# eval_data = 'HepG2_valid_balanced_6mer_6'
# eval_data = "MCF-7_valid_balanced"
# eval_data = 'all_valid_6mer'
# eval_data = 'valid_6mer'
# eval_data = 'test_6mer'
# eval_data = 'CTCF_test_6mer'
dataset = 'Encode3'
# dataset = 'ChIP_690'

# dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/embeddings/'
# files = glob.glob(dataFolder+eval_data+'*pkl')
# files = glob.glob('/new-stg/home/cong/DPI/dataset/**/'+eval_data+'*pkl', recursive=True)
files = glob.glob('/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+eval_data+'*pkl', recursive=True)

print(files)
print(len(files))


def config():
    config = {}
    config['batch_size'] = 32
    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
#     config['max_protein_seq'] = 512
#     config['max_protein_seq'] = 50
    config['max_protein_seq'] = 768
    config["warmup"]  = 5000
    config['files_per_split'] = 8
    config['hid_dim'] = 240
    config['dropout'] = 0.1
    config['lr'] = 1e-2
    config['pf_dim'] = 2048
    config['n_layers'] = 6
    config['n_heads'] = 12
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
        p_v, input_mask_p = unify_dim_embedding(p,self.max_p,return_start_index=False)
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
    
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p)
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
        
#     """Load preprocessed data."""
#     dataset, dataset_size = create_datasets(batch, eval_data)


    """ Load trained model"""
    model = init_model(config)
    file_model = 'output/model/' + model_path
    model.load_state_dict(torch.load(file_model))
    model.to(device)

        
    """Output files."""
    file_AUCs = 'output/result/eval--allTFs-' + dataset+'-'+eval_data + "," + model_path.replace('pt','txt')
    file_labels = 'output/result/eval-allTFs-label--' + dataset + '-' + eval_data+',' + model_path.replace('pt','pkl')

    AUC = ('Split\tTotal_files\tTime(sec)\tLoss_val\taccuracy_dev\tROC_dev\tPRC_dev\tMCC_dev\trecall_dev\tF1_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')
    

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
        
        dataset, dataset_size = create_datasets(batch, frame)
        
        print(f"Split {i+1} Validation ====", flush=True)
        with torch.set_grad_enabled(False):
            model.eval()
            accuracy, ROC_dev, PRC_dev, MCC, recall, f1, loss_dev, y_label, y_pred = predict(dataset,model,use_cuda)
        print(f"GPU memory usage percent: {100*torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}%", flush=True)
        
        end = timeit.default_timer()
        time = end - start

        loss_dev = loss_dev.to('cpu').data.numpy()
        AUCs = [i+1, (i+1)*files_per_split, time, loss_dev, accuracy, ROC_dev, PRC_dev, MCC, recall, f1]
        print('\t'.join(map(str, AUCs))) 

        with open(file_AUCs, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

#         # save pred and true labels (optional) 
#         df = pd.DataFrame({"true": y_label, "pred": y_pred})
#         df.to_pickle(file_labels)      
        
    print('Finished evaluating...')    

