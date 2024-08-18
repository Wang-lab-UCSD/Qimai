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

# dataFolder = '/new-stg/home/cong/DPI/dataset/Encode3/embeddings/'
# input_data = '100_per_exp_train_6mer'  
# eval_data = '100_per_exp_test_6mer'

dataFolder = '/new-stg/home/cong/DPI/dataset/Encode3/embeddings_512bp/'
input_data = '100_per_exp_max_512bp_train_6mer'  
eval_data = '100_per_exp_max_512bp_test_6mer'
# input_data = '100_per_exp_1258exp_RPM_train_norm'  
# eval_data = '100_per_exp_1258exp_RPM_test_norm'
# num = 11


# files = glob.glob(dataFolder+input_data+'*pkl')

# combine ChIP-690 and Encode3 data together
# input_data = '100_per_exp_max_512bp_train_6mer'  
# eval_data = '100_per_exp_max_512bp_test_6mer'
files = glob.glob('/new-stg/home/cong/DPI/dataset/**/'+input_data+'*pkl', recursive=True)
# files.sort()
num = len(files)

# # random select multiple files
# files = random.choices(files, k=num)
# # take first 100 files
# files = files[:num]
print(files)
print(len(files))
    

def config():
    config = {}
    config['batch_size'] = 16
    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
    config['max_protein_seq'] = 512
    config["warmup"]  = 10000
    config['iterations'] = 20
    config['files_per_split'] = 3
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
# variable-length for both dna and protein with upper limit due to memory restriction
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
    seq_length = x.shape[0]
    if seq_length > crop_size:
        start_pos = random.randint(0,seq_length-crop_size)
        x = x[start_pos:start_pos+crop_size]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

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
        d_v = unify_dim_embedding(d,self.max_d)
        
        p = self.df.iloc[index]['protein_embedding']
        p_v = unify_dim_embedding(p,self.max_p)
        y = self.labels[index]
        
#         print(d_v.shape)
#         print(p_v.shape)
#         print(y.shape)
        return d_v, p_v, y

def collate_fn(data):
    '''  
    padding protein and dna embeddings and masks to max length of a batch

    Args:
        data: list of tuple (dna_embedding, protein_embedding, label)
    Return:
        padded_dna - Padded Sequence, tensor of shape (batch_size, padded_dna_length, dna_dim)
        padded_dna_mask - Padded Sequence, tensor of shape (batch_size, padded_dna_length)
        padded_protein - Padded Sequence, tensor of shape (batch_size, padded_protein_length, protein_dim)
        padded_protein_mask - Padded Sequence, tensor of shape (batch_size, padded_protein_length)
        label - tensor of shape (batch_size)
    '''

    # 	data.sort(key=lambda x: len(x[0]), reverse=True)
    dna, pro, label = zip(*data)
    length_dna = [seq.shape[0] for seq in dna]
    dna_dim = dna[0].shape[1]
    length_pro = [seq.shape[0] for seq in pro]
    pro_dim = pro[0].shape[1]

    padding = 0.0
    padded_dna = torch.full((len(dna), max(length_dna), dna_dim), padding)
    padded_dna_mask = torch.full((len(dna), max(length_dna)), padding)
    padded_pro = torch.full((len(pro), max(length_pro), pro_dim), padding)
    padded_pro_mask = torch.full((len(pro), max(length_pro)), padding)

    for i, seq in enumerate(dna):
        end = length_dna[i]
        padded_dna[i,:end] = seq
        padded_dna_mask[i,:end] = 1
    for i, seq in enumerate(pro):
        end = length_pro[i]
        padded_pro[i,:end] = seq
        padded_pro_mask[i,:end] = 1        
#     print(length_dna)
#     print(padded_dna.shape)
#     print(padded_dna_mask.shape)
#     print(length_pro)
#     print(padded_pro.shape)
#     print(padded_pro_mask.shape)
    return padded_dna, padded_pro, padded_dna_mask, padded_pro_mask, torch.from_numpy(np.array(label))

def create_datasets(batch_size, df):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1, 
              'drop_last': True,
              'pin_memory': True,
              'collate_fn': collate_fn}

    
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
    print("training size: ", df_train.shape[0])
    print("validation size: ", df_val.shape[0])
    
    training_set = BIN_Data_Encoder(np.array([i for i in range(df_train.shape[0])]), df_train.label.values, df_train, max_d, max_p)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(np.array([i for i in range(df_val.shape[0])]), df_val.label.values, df_val, max_d, max_p)
    validation_generator = data.DataLoader(validation_set, **params)
    
    return training_generator, validation_generator


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

def test_accuracy(model,config):
    # load test data
    params = {'batch_size': config["batch_size"],
              'shuffle': True,
              'num_workers': 1, 
              'drop_last': True,
              'pin_memory': True}
    df = pd.read_pickle(dataFolder + eval_data + '.pkl')
    # df = pd.read_csv(dataFolder + path + suffix)
    # df = pd.read_pickle(dataFolder + path + '_6_ft.pkl')
    filter = df['dna'].str.contains('N')
    df = df[~filter]
        
    
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, config["max_dna_seq"], config["max_protein_seq"])
    test_data = data.DataLoader(dataset, **params)
    
    # run model 
    # optimizer and scheduler are not needed, just used for place holders
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, 192, factor=1, warmup=4000))    
    use_cuda = torch.cuda.is_available() 
    
    with torch.set_grad_enabled(False):
        model.eval()
        accuracy, AUC_dev, PRC_dev, MCC, recall, f1, loss_dev = run_epoch(
            test_data,
            model,
            optimizer,
            scheduler,
            use_cuda,
            mode="test",
            accum_iter=1
        )
    print('val loss: ', loss_dev)
    return accuracy

        
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
    iterations = config['iterations'] 
    files_per_split = config['files_per_split'] 
    

    """ create model, optimizer and scheduler"""
    model = init_model(config)
    print(model)
    model.to(device)

#     """ Load trained model"""
#     model = init_model(config)
#     file_model = 'output/model/v6-' + 'lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)+'.pt'
#     model.load_state_dict(torch.load(file_model))
#     model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, config['hid_dim'], factor=1, warmup=config["warmup"]))    
    
    """Output files."""
    file_AUCs = 'output/result/AUCs--v7-lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)+'.txt'
    file_model = 'output/model/v7-' + 'lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)+'.pt'
    AUC = ('Epoch\tSplit\tTime(sec)\tLoss_train\tLoss_val\taccuracy_dev\tAUC_dev\tPRC_dev\tMCC_dev\trecall_dev\tF1_dev')

    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

        
    """Start training."""
    print('Training...')
    start = timeit.default_timer()
    max_AUC_dev = 0
    num_split = math.ceil(len(files)/files_per_split)

    for epoch in range(1, iterations + 1):
        
        """Load preprocessed data by chunk"""
        # shuffling the files in every epoch
        random.shuffle(files)

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
            del li
            dataset_train, dataset_dev = create_datasets(batch, frame)

            """Load preprocessed data by chunk"""

            model.train()
            print(f"Epoch {epoch} Split {i+1} Training ====", flush=True)

            loss_train = run_epoch(
                dataset_train,
                model,
                optimizer,
                scheduler,
                use_cuda,
                mode="train",
                accum_iter=1
            )
            GPUtil.showUtilization()
            torch.cuda.empty_cache()


            print(f"Epoch {epoch} Split {i+1} Validation ====", flush=True)
            with torch.set_grad_enabled(False):
                model.eval()
                accuracy, AUC_dev, PRC_dev, MCC, recall, f1, loss_dev = run_epoch(
                    dataset_dev,
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
            AUCs = [epoch,i+1, time, loss_train, loss_dev, accuracy, AUC_dev, PRC_dev, MCC, recall, f1]
            with open(file_AUCs, 'a') as f:
                f.write('\t'.join(map(str, AUCs)) + '\n')

            if AUC_dev > max_AUC_dev:
                torch.save(model.state_dict(), file_model)
                max_AUC_dev = AUC_dev
            print('\t'.join(map(str, AUCs)))
        
    print('Finished training...')    
    
#     """start testing"""
#     print('Testing...')
#     model.load_state_dict(torch.load(file_model))
#     model.to(device)
#     test_acc = test_accuracy(model,config)
#     print(config)
#     print("Test set accuracy: {}".format(test_acc))


