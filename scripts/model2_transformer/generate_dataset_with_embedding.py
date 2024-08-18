import logging
logging.disable(logging.WARNING)
import warnings 
warnings.filterwarnings("ignore")
import torch
import pickle
from transformers import BertModel, BertConfig
from DNABERT_v1.src.transformers import DNATokenizer
import csv
import re
from pathlib import Path
import os
import tarfile
import shutil
import json
import sys
import pandas as pd
import random
import numpy as np
import glob
import math

def get_DNA_embedding(row,kmer):
    sequence = row['dna']
    sequence = ' '.join([sequence[i:i+kmer] for i in range(0, (len(sequence)-kmer+1))])
#     print(sequence)
    with torch.no_grad():
        model_input = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=512, truncation=True)["input_ids"]
        model_input = torch.tensor(model_input, dtype=torch.long)
        model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
#         print(model_input)
        output = model(model_input)

    embed = output[0]
    embed = torch.squeeze(embed)
#     embed = embed.numpy()
#     embed = np.mean(embed, axis=1)
    print("embedding size is "+ str(embed.shape))
#     print("embedding mean is " + str(torch.mean(embed,0)[0:10]))
#     print(embed)
    return embed


def gene_name_to_id(gene_name,mapping_df):
    return mapping_df[mapping_df['name'].str.upper()==gene_name]['id'].tolist()

def pick_single_protein_pickle(fileL):
    """
    pick one file from an unempty list of files
    """
    if len(fileL)>1:
        file = random.sample(fileL,1)[0]
    else:
        file = fileL[0]
    return file

def protein2emb(row):
    pro = row['protein']
    fileL = glob.glob(protein_emb_dir+'*'+pro+'*single_repr*npy', recursive=True)
    if len(fileL)>=1:
        file = pick_single_protein_pickle(fileL)
    # other situations that the file name contains gene synonym
    else:
        idL = gene_name_to_id(pro,mapping_df)
        for id in idL: 
            fileL = glob.glob(protein_emb_dir+'*'+id+'*single_repr*npy', recursive=True)
            if len(fileL) >= 1:
                file = pick_single_protein_pickle(fileL)
            break
    try:
        x = np.load(file)    
        print("protein embedding size is "+str(x.shape))
        return x
    except:
        print(pro,"doesn't have embedding, returns an empty list")
        return None

def add_two_columns(df):
    """
    add dna_embedding and protein_embedding as two columns to df
    """
    col = df.apply(get_DNA_embedding, axis=1, kmer=kmer) 
    df = df.assign(dna_embedding=col.values)
    
    col = df.apply(protein2emb, axis=1) 
    df = df.assign(protein_embedding=col.values)
    df = df.dropna()
#     df[df['protein_embedding'].map(lambda d: len(d)) > 0]
    return df
    
def main(df,output_file):
    """
    write new dataset with embedding
    """
    df = add_two_columns(df)
    df.to_pickle(output_file)
    print("DNA embedding size is "+ str(df.iloc[0]['dna_embedding'].shape))
    print("protein embedding size is "+ str(df.iloc[0]['protein_embedding'].shape))
    

    
    
file = sys.argv[1] 
kmer = int(sys.argv[2])
# start_loc = int(sys.argv[2])
# span = int(sys.argv[3])    
    
# prepare for DNA embedding extraction
dir_to_pretrained_model = '/new-stg/home/cong/DPI/DNABERT/pretrained/'+str(kmer)+'-new-12w-0/'
path_to_config = '/new-stg/home/cong/DPI/DNABERT/src/transformers/dnabert-config/bert-config-'+str(kmer)+'/config.json'
path_to_token = '/new-stg/home/cong/DPI/DNABERT/src/transformers/dnabert-config/bert-config-'+str(kmer)+'/vocab.txt'

# # use fine-tuned model for extraction
# dir_to_pretrained_model = '/new-stg/home/cong/DPI/DNABERT/examples/ft/CTCF_'+str(kmer)+'/'
# path_to_config = '/new-stg/home/cong/DPI/DNABERT/src/transformers/dnabert-config/bert-config-'+str(kmer)+'/config.json'
# path_to_token = '/new-stg/home/cong/DPI/DNABERT/src/transformers/dnabert-config/bert-config-'+str(kmer)+'/vocab.txt'


config = BertConfig.from_pretrained(path_to_config)
tokenizer = DNATokenizer.from_pretrained(path_to_token)
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)

# prepare for dataset loading
protein_emb_dir = '/new-stg/home/cong/DPI/colabfold/output/uniprot_ChIP_690_human_reviewed_162/'
dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/data_s2_by_exp/'
output_folder = '/new-stg/home/cong/DPI/dataset/ChIP_690/data_with_embedding/localColabFold/'

# df_train = pd.read_csv(dataFolder + '/train.csv',header=None,names=['loc','dna','label','protein'],dtype={'loc':str,'dna':str,'label':int,'protein':str})
# df_val = pd.read_csv(dataFolder + '/val.csv',header=None,names=['loc','dna','label','protein'],dtype={'loc':str,'dna':str,'label':int,'protein':str})
# df_test = pd.read_csv(dataFolder + '/test.csv',header=None,names=['loc','dna','label','protein'],dtype={'loc':str,'dna':str,'label':int,'protein':str})


# for later processed files, it already has header, like 'val_v2'
df = pd.read_csv(dataFolder + file + '.csv')
# df = df.iloc[:30]
print(df.shape)

# loading mapping file to convert between gene name and uniprot id just in case
mapping_file = '/new-stg/home/cong/DPI/downloads/HUMAN_9606_idmapping.dat'
mapping_df = pd.read_csv(mapping_file,sep="\t",header=None,names=['id','cat','name'])
mapping_df = mapping_df[mapping_df['cat'].isin(['Gene_Name','Gene_Synonym'])]

# main
random.seed(1)

# # generate one file per 10,000 examples
# span = 10000
# num = int(len(df)/span)
# for i in range(0, num+1):
#     df2 = df.iloc[span*i: min(span*(i+1),len(df))]
#     output_file = output_folder+file+'_'+str(kmer)+'mer_'+str(i+1)+'.pkl'
#     main(df2, output_file)

# generate one file per protein
# if input csv file doesn't have header, like "CTCF_train.csv" or "val.csv"
# df = pd.read_csv(dataFolder + file + '.csv',header=None,names=['loc','dna','label','protein'],dtype={'loc':str,'dna':str,'label':int,'protein':str})
# print(df.shape)
main(df,output_folder+file+'_'+str(kmer)+'mer.pkl')



