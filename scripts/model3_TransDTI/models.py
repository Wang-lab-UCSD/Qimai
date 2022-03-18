from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)


class BIN_Interaction_Flat(nn.Sequential):
    '''
        Interaction Network with 2D interaction map
    '''
    
    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()

        self.dropout_rate = config['dropout_rate']
        self.batch_size = config['batch_size']
        self.gpus = torch.cuda.device_count()

        self.input_dim_dna = config['input_dim_dna']
        self.input_dim_protein = config['input_dim_protein']
        self.emb_dim_dna = config['emb_dim_dna']
        self.emb_dim_protein = config['emb_dim_protein']
        self.flatten_dim = self.emb_dim_protein + self.emb_dim_dna
        
        
        self.demb = FullyConnectedEmbed(self.input_dim_dna, self.emb_dim_dna, self.dropout_rate)
        self.pemb = FullyConnectedEmbed(self.input_dim_protein, self.emb_dim_protein, self.dropout_rate)
        

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        )
        
    def forward(self, d, p):
        """
        :param d: dna embedding (batch_size,max_d,input_dim_dna)
        :type d: torch.Tensor
        :param p: protein embedding (batch_size,max_p,input_dim_protein)
        :type p: torch.Tensor
        """

        d = torch.mean(d,1) # batch_size x input_dim_dna
        p = torch.mean(p,1) # batch_size x input_dim_protein

        d_emb = self.demb(d) # batch_size x emb_dim_dna
        p_emb = self.pemb(p) # batch_size x emb_dim_protein

        f = torch.cat((d_emb,p_emb),1) # batch_size x (emb_dim_dna + emb_dim_protein)
        score = self.decoder(f)
        return score    

   
# help classes    
class FullyConnectedEmbed(nn.Module):
    """
    Takes language model embeddings and outputs low-dimensional embedding instead

    :param nin: size of language model embedding
    :type nin: int
    :param nout: dimension of ouput embedding
    :type nout: int
    :param dropout_rate: portion of weights to drop out [default: 0.5]
    :type droput_rate: float
    :param activation: Activation for linear projection model
    :type activation: torch.nn.Module

    """
    def __init__(self, nin, nout, dropout_rate=0.5, activation=nn.ReLU()):
        super(FullyConnectedEmbed, self).__init__()
        self.nin = nin
        self.nout = nout 
        self.dropout_rate = dropout_rate

        self.l1 = nn.Linear(nin, nout)
        self.bn1 = nn.BatchNorm1d(nout)
        self.drop = nn.Dropout(p=self.dropout_rate)
        self.activation = activation


    def forward(self, x):
        """
        :param x: Input language model embedding
        :type x: torch.Tensor
        :return: low dimensional projection of input embedding
        :rtype: torch.Tensor
        """
        t = self.l1(x)
        t = self.bn1(t)
        t = self.activation(t)
        t = self.drop(t)
        return t
