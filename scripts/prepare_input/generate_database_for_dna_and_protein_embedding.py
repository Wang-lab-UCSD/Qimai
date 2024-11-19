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

random.seed(1)
file = sys.argv[1] # 'test', 'train', 'test'
kmer = int(sys.argv[2]) # 6
dataset = sys.argv[3] # 'Encode3and4' or 'ChIP690'
print('input file: {file}\nkmer: {kmer}\ndataset: {dataset}'.format(file=sys.argv[1] , kmer=sys.argv[2], dataset=sys.argv[3]))


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


    # The first input_id is always 2, which is CLS and the last input_id is always 3 for any sequence, which is SEP based on BERT tradition
    # so we discarded the first and last dim
    embed = embed[1:-1]
#     print("embedding size is "+ str(embed.shape))
#     print("embedding mean is " + str(torch.mean(embed,0)[0:10]))
#     print(embed)
    return embed


def gene_name_to_id(gene_name,mapping_df):
    return mapping_df[mapping_df['name'].str.upper()==gene_name]['id'].tolist()



def protein2emb(row):
    pro = row['protein']
    fileL = glob.glob(protein_emb_dir+'*_'+pro+'_[HUMAN|PE]*single_repr*npy', recursive=True)
    if len(fileL)==0:
        idL = gene_name_to_id(pro,mapping_df)
        for id in idL: 
            fileL = glob.glob(protein_emb_dir+'*_'+id+'_*single_repr*npy', recursive=True)
            if len(fileL)>=1:
                print(pro+': '+id+', had to match with uniprot id.')
                break
    if len(fileL)>=1:
        x = np.mean(np.array([np.load(fileL[i]) for i in range(len(fileL))]), axis=0)
        print("protein: "+pro+". Embedding size is "+str(x.shape))
        return x    
    else:
        print(pro,"doesn't have embedding, returns an empty list")
        return None

    
    
# prepare for DNA embedding extraction
dir_to_pretrained_model = '/new-stg/home/cong/DPI/DNABERT/pretrained/'+str(kmer)+'-new-12w-0/'
path_to_config = '/new-stg/home/cong/DPI/DNABERT/src/transformers/dnabert-config/bert-config-'+str(kmer)+'/config.json'
path_to_token = '/new-stg/home/cong/DPI/DNABERT/src/transformers/dnabert-config/bert-config-'+str(kmer)+'/vocab.txt'

config = BertConfig.from_pretrained(path_to_config)
tokenizer = DNATokenizer.from_pretrained(path_to_token)
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)

# loading mapping file to convert between gene name and uniprot id just in case
mapping_file = '/new-stg/home/cong/DPI/downloads/HUMAN_9606_idmapping.dat'
mapping_df = pd.read_csv(mapping_file,sep="\t",header=None,names=['id','cat','name'])
mapping_df = mapping_df[mapping_df['cat'].isin(['Gene_Name','Gene_Synonym'])]


if dataset == "Encode3":
    protein_emb_dir = "/new-stg/home/cong/DPI/colabfold/output/uniprot-download_Encode3_fasta-2022.11.17-18.19.17.82/"
elif dataset == "Encode4":
    protein_emb_dir = "/new-stg/home/cong/DPI/colabfold/output/uniprot-Encode4_631_ID_641_entry-2023.04.08-18.23.26.46/"
elif dataset == "ChIP_690":
    protein_emb_dir = "/new-stg/home/cong/DPI/colabfold/output/uniprot-download_ChIP690_159ID_163_entry_-2023.04.05-04.53.21.20/"
elif dataset == "Encode3and4":
    protein_emb_dir = "/new-stg/home/cong/DPI/colabfold/output/Encode3and4/"

    
dataFolder = '/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/data/'
output_folder = '/new-stg/home/cong/DPI/dataset/'+dataset+'/deepsea/embeddings/'+file+'/'
Path(output_folder).mkdir(parents=True, exist_ok=True)

### dealing with deepsea output
df = pd.read_pickle(dataFolder + file + '.pkl')
print('total num of rows: '+str(df.shape))

# add new column with index of unique values of another column
df['dna_id'] = df.groupby('dna',sort=False).ngroup()+1



# create database for protein embeddings
# this database stores the mean embedding across multiple models, i.e. each protein only has one embedding
h = df[['protein']].drop_duplicates()
col = h.apply(protein2emb, axis=1) 
h = h.assign(protein_embedding=col.values)
h = h.dropna()
f2 = output_folder+file+'_protein_embedding_mean.pkl'
h.to_pickle(f2)


# create database for dna embeddings
g = df[['dna','dna_id']].drop_duplicates()
span = 5000
total_files = math.ceil(len(g)/span)
print('In total: '+str(total_files)+' files. Each file has '+str(span)+' dna embeddings')
# for i in range(90,total_files):
for i in range(total_files):    
    g2 = g.iloc[span*i: min(span*(i+1),len(g))]
    col = g2.apply(get_DNA_embedding, axis=1, kmer=kmer) 
    g2 = g2.assign(dna_embedding=col.values)
    f1 = output_folder+file+'_dna_embedding_'+str(kmer)+'mer_'+str(i+1)+'.pkl'
    g2.to_pickle(f1)



