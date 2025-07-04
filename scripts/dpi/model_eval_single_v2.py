# for deepsea model evaluation
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
import torch.utils.data as Data
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc, precision_recall_curve, matthews_corrcoef

import numpy as np
import pandas as pd
import random
import os, gc
import glob
import time, timeit
import math
import GPUtil
import copy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from dpi_v0.plot_utils import *
from dpi_v0.main_singletask import *

method = 'DNA'  # Changed from 'TF' to 'DNA'
tag = 'ChIP_690_valid_v2'
get_data = True
return_attention = False
subset_test_pros = False
define_functions = False
random_shuff_on_test_files = True

# Model path
model_path = 'main_singletask_Encode3and4_seed101_800_proteins-lr=0.05,epoch=2,dropout=0.2:0,hid_dim=240,n_layer=2,n_heads=6,batch=128,input=train_min10,max_dna=512,max_protein=768.pt'

# Data paths
data_path = '/new-stg/home/cong/DPI/dataset/transformed_data_ChIP690_valid_v2.pkl' # prepared from ChIP_690 valid data
# pro_emb_path = '/new-stg/home/cong/DPI/dataset/Encode3_Encode4_ChIP690_protein_embedding_mean.pkl'
pro_emb_path = '/new-stg/home/cong/DPI/dataset/all_human_tfs_protein_embedding_mean.pkl'
family_label_path = '/new-stg/home/cong/DPI/dataset/all_TF_family_label.pkl'

# Output paths
output_dir = 'output/result'
os.makedirs(output_dir, exist_ok=True)

def get_config():
    config = {}
    config['batch_size'] = 128  # Changed to match saved model
    config['dna_dim'] = 768
    config['protein_dim'] = 384
    config['max_dna_seq'] = 512
    config['n_family'] = 0
    config['max_protein_seq'] = 768  # Changed to match saved model
    config['warmup'] = 10000
    config['iteration_per_split'] = 2
    config['files_per_split'] = 5
    config['valid_dna_size'] = 500
    config['hid_dim'] = 240
    config['dropout'] = 0.2  # Changed to match saved model
    config['input_dropout'] = 0
    config['lr'] = 0.05  # Changed to match saved model
    config['pf_dim'] = 2048
    config['n_layers'] = 2  # Changed to match saved model
    config['n_heads'] = 6
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['return_attention'] = return_attention
    return config

def init_model(config):
    model = Predictor(**config)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def unify_dim_embedding(x, crop_size, return_start_index=False):
    pad_token_index = 0
    start_pos = 0
    seq_length = x.shape[0]
    if seq_length < crop_size:
        input_mask = ([1] * seq_length) + ([0] * (crop_size - seq_length))
        if x.ndim == 1:
            x = torch.from_numpy(np.pad(x, (0, crop_size - seq_length), 'constant', constant_values=pad_token_index))
        elif x.ndim == 2:
            x = torch.from_numpy(np.pad(x, ((0, crop_size - seq_length),(0,0)), 'constant', constant_values=pad_token_index))
    else:
        start_pos = random.randint(0,seq_length-crop_size)
        x = x[start_pos:start_pos+crop_size]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        input_mask = [1] * crop_size
    return (x, np.asarray(input_mask), start_pos) if return_start_index else (x, np.asarray(input_mask))

class BIN_Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti, max_d, max_p, dna_embs, pro_embs, df_family, return_attention):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.max_d = max_d
        self.max_p = max_p
        self.dna_embs = dna_embs
        self.pro_embs = pro_embs
        self.df_family = df_family
        self.return_start_index = return_attention

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        dna_seq = self.df.iloc[index]['dna']
        d = self.dna_embs.loc[self.dna_embs['dna']==dna_seq,'dna_embedding'].iloc[0]
        d_v, input_mask_d = unify_dim_embedding(d, self.max_d)

        protein = self.df.iloc[index]['protein']
        protein_matches = self.pro_embs.index[self.pro_embs['protein']==protein].tolist()
        if not protein_matches:
            raise ValueError(f"Protein {protein} not found in embeddings")
        protein_idx = protein_matches[0]
        p = torch.from_numpy(self.pro_embs.loc[protein_idx,'protein_embedding'])
        p_outs = unify_dim_embedding(p, self.max_p, self.return_start_index)
        p_v, input_mask_p = p_outs[0], p_outs[1]
        y = self.labels[index]

        return (d_v, p_v, input_mask_d, input_mask_p, y, dna_seq, protein) if not self.return_start_index else (d_v, p_v, input_mask_d, input_mask_p, y, p_outs[2], dna_seq, protein)

def create_datasets(batch_size, max_d, max_p, df, dna_embs, pro_embs, df_family, return_attention):
    # Filter for DNA sequences that have embeddings
    dna_seqs = dna_embs.dna.unique().tolist()
    df = df.loc[df.dna.isin(dna_seqs)]
    
    # Filter for proteins that have embeddings
    protein_seqs = pro_embs.protein.unique().tolist()
    df = df.loc[df.protein.isin(protein_seqs)]
    
    if len(df) == 0:
        raise ValueError("No valid examples found after filtering for DNA and protein embeddings")

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2,  # Reduced from 4 to 2
              'drop_last': False,
              'pin_memory': True}

    dataset = BIN_Data_Encoder(np.array([i for i in range(df.shape[0])]), df.label.values, df, max_d, max_p, dna_embs, pro_embs, df_family, return_attention)
    generator = data.DataLoader(dataset, **params)

    return generator, len(dataset)

def load_dna_embeddings_batch(dna_emb_dir, target_dna_seqs, batch_size=1):  # Reduced from 3 to 1
    """Load DNA embeddings in batches to save memory, only keeping embeddings for target DNA sequences"""
    print(f"\nLoading DNA embeddings from: {dna_emb_dir}")
    dna_emb_files = sorted(glob.glob(dna_emb_dir + 'valid_dna_embedding_6mer_*.pkl'))
    print(f"Found {len(dna_emb_files)} DNA embedding files")
    target_dna_set = set(target_dna_seqs)
    print(f"Target DNA sequences: {len(target_dna_set)}")
    
    for i in range(0, len(dna_emb_files), batch_size):
        batch_files = dna_emb_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} with {len(batch_files)} files")
        dna_emb_list = []
        for file in batch_files:
            print(f"Loading file: {file}")
            df = pd.read_pickle(file)
            print(f"File contains {len(df)} DNA sequences")
            # Filter only target DNA sequences
            df = df[df['dna'].isin(target_dna_set)]
            print(f"After filtering: {len(df)} DNA sequences")
            if not df.empty:
                dna_emb_list.append(df)
        
        if dna_emb_list:
            batch_df = pd.concat(dna_emb_list, axis=0, ignore_index=True)
            print(f"Batch contains {len(batch_df)} DNA sequences")
            yield batch_df
            del dna_emb_list
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("No matching DNA sequences found in this batch")

def evaluate_dna_sequence(dna_seq, data_dna, model, device, protein_graph_node_ft, batch_size, max_d, max_p, dna_embs, pro_embs, df_family):
    print(f"\nEvaluating DNA sequence: {dna_seq}")
    print(f"Number of examples for this DNA: {len(data_dna)}")
    
    try:
        df, df_size = create_datasets(batch_size, max_d, max_p, data_dna, dna_embs, pro_embs, df_family, return_attention)
        print(f"Created dataset with {df_size} examples")
        
        with torch.set_grad_enabled(False):
            model.eval()
            predictions = []
            for batch in df:
                if return_attention:
                    dna_emb, pro_emb, input_mask_d, input_mask_p, labels, start_pos, dna_seqs, proteins = batch
                else:
                    dna_emb, pro_emb, input_mask_d, input_mask_p, labels, dna_seqs, proteins = batch
                
                dna_emb = dna_emb.to(device)
                pro_emb = pro_emb.to(device)
                labels = labels.to(device)
                input_mask_d = input_mask_d.to(device).float()  # Convert to float
                input_mask_p = input_mask_p.to(device).float()  # Convert to float
                
                # Format masks as required by the model
                dna_mask = input_mask_d.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_d]
                protein_mask = input_mask_p.unsqueeze(1).unsqueeze(3)  # [batch, 1, max_p, 1]
                cross_attn_mask = torch.matmul(protein_mask, dna_mask)  # [batch, 1, max_p, max_d]
                dna_mask = torch.matmul(dna_mask.permute(0,1,3,2), dna_mask)  # [batch, 1, max_d, max_d]
                protein_mask = torch.matmul(protein_mask, protein_mask.permute(0,1,3,2))  # [batch, 1, max_p, max_p]
                
                outputs = model(dna_emb, pro_emb, dna_mask, protein_mask, cross_attn_mask)
                pred_labels = torch.sigmoid(outputs).cpu().numpy()
                true_labels = labels.cpu().numpy()
                
                # Store results
                for pred, true, seq, prot in zip(pred_labels, true_labels, dna_seqs, proteins):
                    predictions.append({
                        'pred_label': float(pred),  # Convert to float scalar
                        'true_label': float(true),  # Convert to float scalar
                        'dna_seq': seq,
                        'protein': prot
                    })
            
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            y_preds = pred_df['pred_label'].values
            y_labels = pred_df['true_label'].values
            
            accuracy, sensitivity, specificity, ROC, PRC, MCC, F1 = evaluate(y_preds, y_labels)
            print(f"Results - Accuracy: {accuracy:.4f}, ROC: {ROC:.4f}, PRC: {PRC:.4f}")
            return {
                'dna': dna_seq,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ROC': ROC,
                'PRC': PRC,
                'MCC': MCC,
                'F1': F1,
                'predictions': pred_df
            }
    except ValueError as e:
        print(f"Warning: {str(e)}")
        return None

def main():
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load config and model
    config = get_config()
    model = init_model(config)
    print(f"Loading model from: output/model/{model_path}")
    model.load_state_dict(torch.load(f'output/model/{model_path}'))
    model.to(device)
    
    # Load data
    print('Loading data...')
    dpi_data = pd.read_pickle(data_path)
    print(f"Loaded DPI data with {len(dpi_data)} examples")
    pro_embs = pd.read_pickle(pro_emb_path).reset_index()
    print(f"Loaded protein embeddings for {len(pro_embs)} proteins")
    df_family = pd.read_pickle(family_label_path)
    
    # Get target DNA sequences
    target_dna_seqs = dpi_data.dna.unique()
    print(f'Found {len(target_dna_seqs)} target DNA sequences')
    
    # Create protein graph node features
    protein_graph_node_ft = torch.from_numpy(np.stack(pro_embs.apply(lambda row: np.mean(row['protein_embedding'], axis=0), axis=1)))
    
    # Setup output files with correct paths
    output_dir = 'output/result'
    os.makedirs(output_dir, exist_ok=True)
    file_AUCs = os.path.join(output_dir, f'eval--DNA_{tag}{method},{model_path.replace("pt","txt")}')
    file_pred_dfs = os.path.join(output_dir, f'pred_df--DNA_{tag}{method},{model_path.replace("pt","pkl")}')
    
    print(f"\nWriting results to: {file_AUCs}")
    # Write header
    with open(file_AUCs, 'w') as f:
        f.write('dna\taccuracy\tsensitivity\tspecificity\tROC\tPRC\tMCC\tF1\n')
    
    # Evaluate per DNA sequence
    print('Starting evaluation...')
    start = timeit.default_timer()
    
    print(f'In total: {len(pro_embs.protein.unique())} proteins; {len(target_dna_seqs)} dna sequences; {len(dpi_data)} examples are included in testing.')
    
    output_df = []
    dna_emb_dir = '/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/embeddings/valid/'
    
    # Process DNA embeddings in batches
    for batch_idx, dna_embs in enumerate(load_dna_embeddings_batch(dna_emb_dir, target_dna_seqs, batch_size=1)):  # Reduced from 3 to 1
        print(f'\nProcessing DNA embedding batch {batch_idx+1}')
        
        # Get DNA sequences in current batch
        batch_dna_seqs = dna_embs.dna.unique()
        batch_dna_seqs = [seq for seq in target_dna_seqs if seq in batch_dna_seqs]
        print(f"Found {len(batch_dna_seqs)} DNA sequences to evaluate in this batch")
        
        if not batch_dna_seqs:
            print("No DNA sequences to evaluate in this batch, skipping...")
            continue
            
        # Process one DNA sequence at a time
        for dna_seq in batch_dna_seqs:
            data_dna = dpi_data[dpi_data.dna == dna_seq]
            print(f"\nDNA sequence {dna_seq} has {len(data_dna)} examples before filtering")
            
            # Print memory usage before processing
            print("\nMemory usage before processing:")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
            
            result = evaluate_dna_sequence(dna_seq, data_dna, model, device, protein_graph_node_ft,
                                         config['batch_size'], config['max_dna_seq'],
                                         config['max_protein_seq'], dna_embs, pro_embs, df_family)
            
            if result is None:
                continue
                
            dna_seq = result['dna']
            metrics = [dna_seq, result['accuracy'], result['sensitivity'], result['specificity'],
                      result['ROC'], result['PRC'], result['MCC'], result['F1']]
            
            print(f"Writing results for DNA: {dna_seq}")
            with open(file_AUCs, 'a') as f:
                f.write('\t'.join(map(str, metrics)) + '\n')
            
            if get_data:
                output_df.append(result['predictions'])
            
            # Clear memory after each DNA sequence
            gc.collect()
            torch.cuda.empty_cache()
            
            # Print memory usage after processing
            print("\nMemory usage after processing:")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        
        # Clean up memory
        del dna_embs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save predictions if requested
    if get_data and output_df:
        print("\nSaving predictions...")
        final_output = pd.concat(output_df, axis=0, ignore_index=True)
        final_output.to_pickle(file_pred_dfs)
        print(f"Saved predictions to: {file_pred_dfs}")
        del final_output
        gc.collect()
    
    end = timeit.default_timer()
    time = end - start
    print('\nFinished evaluating...')
    print(f'Total elapsed time: {round(time/3600,2)}h')
    
    # Plot results
    print("\nGenerating plots...")
    sns.set(font_scale=1.5)
    sns.set_style(style='ticks')
    df = get_metric_df(file_AUCs)
    get_metric_plot(df, file_AUCs)
    print("Plots generated successfully")

if __name__ == "__main__":
    main() 