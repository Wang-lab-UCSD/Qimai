# we used DNABERT2 instead
import logging
logging.disable(logging.WARNING)
import warnings 
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModel
import torch
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
import lmdb
import pickle
import gc

random.seed(1)
file = sys.argv[1] # 'test', 'train', 'test'
dataset = sys.argv[2] # 'Encode3and4' or 'ChIP690'
print('input file: {file}\ndataset: {dataset}'.format(file=sys.argv[1], dataset=sys.argv[2]))


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load DNABERT2 model and tokenizer
model_name = "zhihan1996/DNABERT-2-117M"  
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    
def process_dna_batch(sequences, batch_size=128, max_length=512):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors = 'pt', max_length = max_length, padding='max_length', truncation=True)["input_ids"].to(device)
        with torch.no_grad():
            batch_embeddings = model(inputs)[0] # [batch_size, sequence_length, 768]
            # print(f"embedding size: {batch_embeddings.shape}")
        embeddings.extend(batch_embeddings.cpu().numpy())
    return embeddings


def save_embeddings_to_lmdb(embeddings, output_file):
    map_size = 1024 * 1024 * 1024 * 1024  # 1TB, adjust as needed
    env = lmdb.open(output_file, map_size=map_size)
    with env.begin(write=True) as txn:
        for key, value in embeddings.items():
            txn.put(key.encode(), pickle.dumps(value))
    env.close()


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

# add new column with index of unique values of another column
df['dna_id'] = df.groupby('dna',sort=False).ngroup()+1
df[['dna_id', 'protein', 'label']].to_csv(dataFolder + file + '_id.csv')
print('total num of rows: '+str(len(df)))

df2 = df[['dna','dna_id']].drop_duplicates()
print('total num of dna sequences: '+str(len(df2)))


# Process in chunks
chunk_size = 50000  # Adjust based on your RAM capacity
num_chunks = math.ceil(len(df2) / chunk_size)

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df2))
    
    chunk_df = df2.iloc[start_idx:end_idx]
    
    # Process DNA sequences in this chunk
    dna_sequences = chunk_df['dna'].tolist()
    dna_ids = chunk_df['dna_id'].tolist()
    embeddings = process_dna_batch(dna_sequences)
    

    # Create a dictionary of embeddings
    embedding_dict = {}
    for dna_id, embedding in zip(dna_ids, embeddings):
        embedding_dict[str(dna_id)] = embedding
    
    # Save to LMDB
    output_file = f"{output_folder}{file}_dnabert2_embedding_512_chunk_{i+1}.lmdb"
    save_embeddings_to_lmdb(embedding_dict, output_file)
    
    print(f"Processed and saved chunk {i+1}/{num_chunks}")

    # Clear memory
    del chunk_df, dna_sequences, embeddings, embedding_dict
    gc.collect()
    torch.cuda.empty_cache()




