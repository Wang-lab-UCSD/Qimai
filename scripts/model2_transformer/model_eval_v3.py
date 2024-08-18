import torch
from torch.utils import data
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import glob
import time
import math
from scipy import stats

from models_v2 import *
import timeit
import pickle



dataFolder = '/new-stg/home/cong/DPI/dataset/Encode3/embeddings_512bp/'
# input_data = '100_per_exp_max_512bp_train_6mer'  
# input_data = '100_per_exp_1258exp_RPM_train_norm'  
# input_data = 'ENCFF574UKD_POLR2A_GRCh38_norm'
# input_data = 'ENCFF803RIY_CTCF_GRCh38_6mer'
# input_data = 'ENCFF803RIY_CTCF_GRCh38_norm'
# input_data = 'ENCFF584CZG_CTCF_GRCh38_norm'
# input_data = 'ENCFF*CTCF_GRCh38_norm'


# input_data = 'ENCFF'
# eval_data = 'ENCFF803RIY_CTCF_GRCh38_norm_6mer_9'
# eval_data = 'ENCFF584CZG_CTCF_GRCh38_norm_6mer_4'
# eval_data = 'ENCFF584CZG_CTCF_GRCh38_norm'
# eval_data = 'ENCFF803RIY_CTCF_GRCh38_norm'
# eval_data = '100_per_exp_1258exp_RPM_test_norm'
# eval_data = 'ENCFF186NOM_CTCF_GRCh38_norm'
eval_data = 'ENCFF803RIY_CTCF_GRCh38_norm_no_negative'

# list all input data
# files = glob.glob(dataFolder+input_data+'*pkl')
# num = len(files)
# num = 11

# model_path = "v5-qtnorm-lr=0.05,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=100_per_exp_1258exp_RPM_train_norm,num_files=11,max_dna=512,max_protein=512.pt"
# model_path = "v5-lr=0.05,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=ENCFF*CTCF_GRCh38_norm,num_files=19,max_dna=512,max_protein=512.pt"
model_path = "v5-lr=0.05,dropout=0.1,hid_dim=240,n_layer=6,batch=16,input=ENCFF803RIY_CTCF_GRCh38_norm_no_negative,num_files=6,max_dna=512,max_protein=512.pt"


# list all evaluation data
files = glob.glob(dataFolder+eval_data+'*pkl')
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
    config['iteration_per_split'] = 20
    config['files_per_split'] = 3
    config['hid_dim'] = 240
    config['dropout'] = 0.1
    config['lr'] = 5e-2
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
        return d_v, p_v, input_mask_d, input_mask_p, y


def create_datasets(batch_size, df):
    print('--- Data Preparation ---')

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1, 
          'drop_last': True,
          'pin_memory': True}
        
    print("dataset size: ", df.shape[0])
    
    # split into train and val and test on val
    _, df = train_test_split(df, test_size=0.2, random_state=42)
    
    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p)
#     dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.log.values, df, max_d, max_p)
#     dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.qtnorm.values, df, max_d, max_p)

    
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
        
#     """Load preprocessed data."""
#     dataset, dataset_size = create_datasets(batch, eval_data)


    """ Load trained model"""
    model = init_model(config)
#     file_model = 'output/model/v5-' + 'lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)+'.pt'
    file_model = 'output/model/' + model_path
    model.load_state_dict(torch.load(file_model))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, config['hid_dim'], factor=1, warmup=config["warmup"]))    
        
        
    """Output files."""
#     file_AUCs = 'output/result/eval--v5='+eval_data+',AUCs--lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)+'.txt'

#     file_labels = 'output/result/eval-label--v5='+eval_data+',AUCs--lr='+str(lr)+',dropout='+str(dropout)+',hid_dim='+str(hid_dim)+',n_layer='+str(n_layers)+',batch='+str(batch)+',input='+input_data+',num_files='+str(num)+',max_dna='+str(max_d)+',max_protein='+str(max_p)+'.txt'

    file_AUCs = 'output/result/eval--' + eval_data + model_path.replace('pt','txt')
    file_labels = 'output/result/eval-label--' + eval_data + model_path.replace('pt','txt')

    AUC = ('Split\tTotal_files\tProtein\tTime(sec)\tLoss_val\tr2_dev\tMSE_dev\tRMSE_dev\tMAE_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')
    with open(file_labels, 'w') as f:
        f.write('true\tpred\n')
        
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
        del li
        """Load preprocessed data"""
        for x, y in frame.groupby('protein'):
            if y.shape[0] >= batch:
                dataset, dataset_size = create_datasets(batch, y)

                print(f"Split {i+1} Validation ====", flush=True)
                with torch.set_grad_enabled(False):
                    model.eval()
                    r2, MSE, RMSE, MAE, loss_dev, y_label, y_pred = run_epoch(
                        dataset,
                        model,
                        optimizer,
                        scheduler,
                        use_cuda,
                        mode="test",
                        accum_iter=1
                    )
                end = timeit.default_timer()
                time = end - start

                loss_dev = loss_dev.to('cpu').data.numpy()
            #         train_loss.append(loss_train)
            #         val_loss.append(loss_dev)
            #         val_auc.append(AUC_dev)
            #         val_accuracy.append(accuracy)
                AUCs = [i+1, (i+1)*files_per_split, x, time, loss_dev, r2, MSE, RMSE, MAE]
                print('\t'.join(map(str, AUCs))) 

                with open(file_AUCs, 'a') as f:
                    f.write('\t'.join(map(str, AUCs)) + '\n')
                    
                df = pd.DataFrame({"true": y_label, "pred": y_pred})
                df.to_csv(file_labels, header=None, index=None, sep = "\t", mode="a")
    
            else:
                print(x+" only has "+str(y.shape[0])+" samples.")
        
    
    # scatter plot to show pearson correlation
    sns.set(font_scale=1.5)
    sns.set_style(style='ticks')
    plt.rcParams['figure.figsize'] = [6,5]
    df = pd.read_csv(file_labels, sep = "\t")
    graph = sns.jointplot(x= "true", y = "pred", data=df, kind="reg", scatter_kws={"s": 10})
    r, p = stats.pearsonr(df.true, df.pred)
    phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)

    graph.ax_joint.legend([phantom],['Pearson_r={:.3f}'.format(r)])
#     graph.ax_joint.legend([phantom],['Pearson_r={:.3f}, data={}'.format(r,df.shape[0])])
    plt.savefig(file_labels.replace('txt','pdf'))

    print('Finished evaluating...')    

