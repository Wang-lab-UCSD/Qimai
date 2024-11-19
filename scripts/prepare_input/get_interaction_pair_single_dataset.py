import h5py
import numpy as np
import pandas as pd
import sys
import random

nucleotide_dict = {0: 'A', 1: 'G', 2: 'C', 3: 'T'}

# read output from the deepsea pipeline
def read_data(fp,prefix):
    f = h5py.File(fp, "r")
    print(list(f.keys()))
    y = f[prefix+'data'][()]
    x = f[prefix+'xdata'][()]
    return x, y


# Define a function to convert one-hot encoding to DNA sequence
def one_hot_to_sequence(one_hot):
    # Convert numpy array to integer indices
    indices = np.argmax(one_hot, axis=1)
    # Map integer indices to nucleotides using the nucleotide dictionary
    nucleotides = [nucleotide_dict[i] for i in indices]
    # Join nucleotides to form the original DNA sequence
    sequence = ''.join(nucleotides)
    return sequence

# get label for each TF
## for ChIP690 dataset
# def get_label_each_TF(TF,labels,meta):
#     ids = meta.index[meta["TF"]==TF].tolist()
#     if len(ids)==1:
#         return labels[ids[0]]
#     else:
#         no, yes = 0, 0
#         for i in ids:
#             if labels[i]==0:
#                 no +=1
#             else:
#                 yes+=1
# #         print("number of 0s for "+TF+": "+str(no))
# #         print("number of 1s for "+TF+": "+str(yes))
#         if yes>=no:
#             return 1
#         else:
#             return 0

# def get_interaction_label(ylabel, meta, is_balance=False):
#     TF_list = meta.TF.unique().tolist()
#     y = [get_label_each_TF(x, ylabel, meta) for x in TF_list]
# #     for i in range(len(y)):
# #         print("Label for "+TF_list[i]+": "+str(y[i]))
#     # Count the number of zeroes and ones in the list
#     num_zeroes = y.count(0)
#     num_ones = y.count(1)
# #     print('number of zeroes: '+str(num_zeroes))
# #     print('number of ones: '+str(num_ones))
    
#     if is_balance:        
#         # Check if there are more ones than zeroes
#         if num_ones <= num_zeroes:
#             # Find the positions of all ones in the list
#             one_positions = [i for i, val in enumerate(y) if val == 1]

#             # Randomly sample the same number of zeroes as ones
#             zero_positions = random.sample([i for i, val in enumerate(y) if val == 0], k=len(one_positions))
            
#             combined = one_positions + zero_positions
#             y = [y[i] for i in combined]
#             TF_list = [TF_list[i] for i in combined]
#         else:
#             print("Warning: Number of ones is greater than number of zeroes.")       
        
#     return TF_list, y       

## for Encode3 and Encode4 datasets
def get_label_each_TF(TF,labels,meta):
    ids = meta.index[meta['Experiment target']==TF].tolist()
    if len(ids)==1:
        return labels[ids[0]]
    else:
        no, yes = 0, 0
        for i in ids:
            if labels[i]==0:
                no +=1
            else:
                yes+=1
#         print("number of 0s for "+TF+": "+str(no))
#         print("number of 1s for "+TF+": "+str(yes))
        if yes>=no:
            return 1
        else:
            return 0

        
def get_interaction_label(ylabel, meta, is_balance=False):
    TF_list = [element.replace("-human", "") for element in meta['Experiment target'].unique().tolist()]
    y = [get_label_each_TF(x, ylabel, meta) for x in TF_list]
#     for i in range(len(y)):
#         print("Label for "+TF_list[i]+": "+str(y[i]))
    # Count the number of zeroes and ones in the list
    num_zeroes = y.count(0)
    num_ones = y.count(1)
#     print('number of zeroes: '+str(num_zeroes))
#     print('number of ones: '+str(num_ones))
    
    if is_balance:        
        # Check if there are more ones than zeroes
        if num_ones <= num_zeroes:
            # Find the positions of all ones in the list
            one_positions = [i for i, val in enumerate(y) if val == 1]

            # Randomly sample the same number of zeroes as ones
            zero_positions = random.sample([i for i, val in enumerate(y) if val == 0], k=len(one_positions))
            
            combined = one_positions + zero_positions
            y = [y[i] for i in combined]
            TF_list = [TF_list[i] for i in combined]
        else:
            print("Warning: Number of ones is greater than number of zeroes.")       
        
    return TF_list, y        

# get dataframe with three columns: dna, label, protein
def get_interaction_df(x, y, metadata_s):

    Ds, Ts, Ys = [],[],[]
   
    for i in range(len(y)):
        TF_list, y1 = get_interaction_label(y[i], metadata_s, is_balance=True)
        if sum(y1)>=1:
    #     if sum(y1)>=0:
            Ds.extend([one_hot_to_sequence(x[i]) for k in range(len(y1))])
            Ys.extend(y1)
            Ts.extend(TF_list)
    df = pd.DataFrame({'dna': Ds, 'label': Ys, 'protein': Ts})

    return df


# dataFolder = "/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/"
# dataFolder1 = "/new-stg/home/cong/DPI/dataset/Encode3/deepsea/"
# dataFolder2 = "/new-stg/home/cong/DPI/dataset/Encode4/deepsea/"
dataFolder = "/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/"

# metadata = pd.read_csv(dataFolder + "690_ChIP_seq_hg19_TF.csv", index_col=0)
# metadata1 = pd.read_csv("/new-stg/home/cong/DPI/downloads/metadata_Encode3_20230405.tsv", delimiter="\t")
# metadata2 = pd.read_csv("/new-stg/home/cong/DPI/downloads/Encode4_metadata.tsv", delimiter="\t")
metadata = pd.read_csv("/new-stg/home/cong/DPI/downloads/metadata_Encode3and4_20230928.tsv", delimiter="\t")

# cellType = "K562"
# cellType = sys.argv[1]
# if cellType=="all":
#     metadata_s = metadata
# else:
#     metadata_s = metadata.loc[metadata["Cell"]==cellType]

# fileName = "test"
# fileName = sys.argv[2] 
fileName = sys.argv[1]
print(fileName)

x, y = read_data(fp = dataFolder+"data/"+fileName+".mat", prefix=fileName)
print("loaded data")
print(x.shape)
print(y.shape)

Ds, Ts, Ys = [],[],[]
for i in range(len(y)):
    TF_list, y1 = get_interaction_label(y[i], metadata, is_balance=False)
    if sum(y1)>=1:
#     if sum(y1)>=0:
        Ds.extend([one_hot_to_sequence(x[i]) for k in range(len(y1))])
        Ys.extend(y1)
        Ts.extend(TF_list)
        
        
df = pd.DataFrame({'dna': Ds, 'label': Ys, 'protein': Ts})
df.head()
print(len(df))
df.to_pickle(dataFolder+"data/"+fileName+".pkl")


# # combine Encode3 and Encode4
# df = pd.concat([df1, df2])
# print(len(df))
# df.to_pickle("/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/"+fileName+"_s"+str(i)+".pkl")
# del df1, df2


# sampled_df = df.sample(frac=0.1, random_state=42)  # Set random_state for reproducibility
# sampled_df.to_pickle(dataFolder1+"data/"+fileName+"_10pct.pkl")

# # df.to_pickle(dataFolder+"data/"+cellType+"_"+fileName+".pkl")
# # df.to_pickle(dataFolder+"data/"+cellType+"_"+fileName+"_balanced.pkl")
# # df.to_pickle(dataFolder+"data/"+cellType+"_"+fileName+"_true_nega.pkl")

# # # subset to specific TF
# # protein = "CTCF"
# # df = df[df.protein==protein]
# # print(len(df))
# # df.to_pickle(dataFolder+"data/"+cellType+"_"+protein+"_"+fileName+".pkl")
