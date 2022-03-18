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
        input_mask = ([1] * seq_length) + ([0] * (crop_size - seq_length))
        if x.ndim == 1:
            x = np.pad(x, (0, crop_size - seq_length), 'constant', constant_values = 0)
        elif x.ndim == 2:
            x = np.pad(x, ((0, crop_size - seq_length),(0,0)), 'constant', constant_values = 0)
    else:
        start_pos = random.randint(0,seq_length-crop_size)
        x = x[start_pos:start_pos+crop_size]
        input_mask = [1] * crop_size
    return x, np.asarray(input_mask)


# def protein2emb_encoder(index,max_p):
#     """
#     loads protein embedding given row index of data df
    
#     :param index: row index 
#     :type idndex: int
#     :param max_p: crop size for protein 
#     :type max_p: int
#     :return x: unified ColabFold output embedding (crop_size,)
#     :type x: numpy.ndarray
    
#     """
#     x = df.iloc[index]['protein_embedding']

#     return unify_dim_embedding(x,max_p)

# def dna2emb_encoder(index,max_d):
#     """
#     loads protein embedding given row index of data df
    
#     :param index: row index 
#     :type idndex: int
#     :param max_d: crop size for dna 
#     :type max_d: int
#     :return x: unified ColabFold output embedding (crop_size,)
#     :type x: numpy.ndarray
    
#     """
#     x = df.iloc[index]['dna_embedding']

#     return unify_dim_embedding(x,max_d)



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
        p = self.df.iloc[index]['protein_embedding']
        d_v, input_mask_d = unify_dim_embedding(d,self.max_d)
        p_v, input_mask_p = unify_dim_embedding(p,self.max_p)
        
        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y