import numpy as np
import pandas as pd
import torch
from torch.utils import data
import random


max_d = 19
max_p = 512
 

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
    if seq_length < crop_size:
        if x.ndim == 1:
            x = np.pad(x, (0, crop_size - seq_length), 'constant', constant_values = 0)
        elif x.ndim == 2:
            x = np.pad(x, ((0, crop_size - seq_length),(0,0)), 'constant', constant_values = 0)
    else:
        start_pos = random.randint(0,seq_length-crop_size)
        x = x[start_pos:start_pos+crop_size]
    return x


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        d = self.df.iloc[index]['dna_embedding']
        p = self.df.iloc[index]['protein_embedding']
        d_v = unify_dim_embedding(d,max_d)
        p_v = unify_dim_embedding(p,max_p)

        y = self.labels[index]
        return d_v, p_v, y