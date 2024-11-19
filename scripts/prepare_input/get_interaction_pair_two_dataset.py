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
def get_label_each_TF(TF,labels,meta):
    ids = meta.index[meta['TF']==TF].tolist()
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
    TF_list = meta.TF.unique().tolist()
    y = [get_label_each_TF(x, ylabel, meta) for x in TF_list]
#     for i in range(len(y)):
#         print("Label for "+TF_list[i]+": "+str(y[i]))

    
    if is_balance:
        # Count the number of zeroes and ones in the list
        num_zeroes = y.count(0)
        num_ones = y.count(1)
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
            print('number of zeroes: '+str(num_zeroes))
            print('number of ones: '+str(num_ones))
            print("Warning: Number of ones is greater than number of zeroes.")       
            for i in range(len(y)):
                print("Label for "+TF_list[i]+": "+str(y[i]))        
    return TF_list, y        

# get dataframe with three columns: dna, label, protein
def get_interaction_df(x, y, metadata_s, is_balance, min_pair_per_dna=2):

    Ds, Ts, Ys = [],[],[]
   
    for i in range(len(y)):
        TF_list, y1 = get_interaction_label(y[i], metadata_s, is_balance=is_balance)
        if (sum(y1)>=1) & (len(y1)>=min_pair_per_dna):
    #     if sum(y1)>=0:
            Ds.extend([one_hot_to_sequence(x[i]) for k in range(len(y1))])
            Ys.extend(y1)
            Ts.extend(TF_list)
    df = pd.DataFrame({'dna': Ds, 'label': Ys, 'protein': Ts})

    return df



# # cellType = "K562"
# cellType = sys.argv[1]
# if cellType=="all":
#     metadata_s = metadata
# else:
#     metadata_s = metadata.loc[metadata["Cell"]==cellType]

dataset = sys.argv[1] 
# fileName = "test"
fileName = sys.argv[2] 
# i = int(sys.argv[2])
print(fileName)
# print('split '+str(i))

dataFolder = "/new-stg/home/cong/DPI/dataset/"+dataset+"/deepsea/data/"

if dataset == "Encode3":
    metadata = pd.read_csv("/new-stg/home/cong/DPI/downloads/metadata_Encode3_20230405.tsv", delimiter="\t")
    metadata['TF'] = [element.replace("-human", "") for element in metadata['Experiment target'].tolist()]
elif dataset == "Encode4":
    metadata = pd.read_csv("/new-stg/home/cong/DPI/downloads/Encode4_metadata.tsv", delimiter="\t")
    metadata['TF'] = [element.replace("-human", "") for element in metadata['Experiment target'].tolist()]
elif dataset == "Encode3and4":
    metadata = pd.read_csv("/new-stg/home/cong/DPI/downloads/metadata_Encode3and4_20230928.tsv", delimiter="\t")
    metadata['TF'] = [element.replace("-human", "") for element in metadata['Experiment target'].tolist()]
elif dataset == "ChIP_690":
    metadata = pd.read_csv("/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/690_ChIP_seq_hg19_TF.csv", index_col=0)


x, y = read_data(fp = dataFolder+fileName+".mat", prefix=fileName)
print("loaded data "+dataset+":")
print(x.shape) # each element in x is each DNA sequence
print(y.shape) # each element in y is a vector, size is number of proteins

# # output data is much smaller
# is_balance = True
# df = get_interaction_df(x, y, metadata, is_balance)
# df.to_pickle(dataFolder+fileName+".pkl")  

# output df is very large, exceeds the memory limit
# need to do it in chunks
# is_balance = True
is_balance = False
# min_pair_per_dna = 10
min_pair_per_dna = 1
# i = 0 

# span = 250000
# x1 = x[span*i: min(span*(i+1),len(x))]
# y1 = y[span*i: min(span*(i+1),len(y))]
# df = get_interaction_df(x1, y1, metadata, is_balance)
# df.to_pickle(dataFolder+'all_'+fileName+'_s'+str(i)+".pkl") # is_balance=False

df = get_interaction_df(x, y, metadata, is_balance, min_pair_per_dna)
dpi_w = df.pivot(index='dna', columns='protein', values='label')
dpi_w.head()
print(dpi_w.sum(axis=0))
dpi_w.to_pickle(dataFolder+'all_'+fileName+"_wide.pkl") # is_balance=False

# df.to_pickle(dataFolder+fileName+'_min'+str(min_pair_per_dna)+'.pkl')  # is_balance=True
# df.to_pickle(dataFolder+'all_'+fileName+".pkl") # is_balance=False

print('balanced dataset: '+str(is_balance))
print('minimum pair per dna: '+str(min_pair_per_dna))
print('# of examples: '+str(df.shape[0]))
print("success!")
# df1.to_pickle(dataFolder1+fileName+'_s'+str(i)+".pkl")


# # # 1. read Encode3 data 
# x1, y1 = read_data(fp = dataFolder1+fileName+".mat", prefix=fileName)
# print("loaded data")
# print(x1.shape)
# print(y1.shape)

# # x1 = x1[span*i: min(span*(i+1),len(x1))]
# # y1 = y1[span*i: min(span*(i+1),len(y1))]

# df1 = get_interaction_df(x1, y1, metadata1)
# df1.to_pickle(dataFolder1+fileName+".pkl")
# # df1.to_pickle(dataFolder1+fileName+'_s'+str(i)+".pkl")

# # # sampled_df1 = df1.sample(frac=0.1, random_state=42)  # Set random_state for reproducibility
# # # sampled_df1.to_pickle(dataFolder1+"data/"+fileName+'_s'+str(i)+"_10pct.pkl")


# # 2. read Encode4 data 
# x2, y2 = read_data(fp = dataFolder2+fileName+".mat", prefix=fileName)
# print("loaded data")
# print(x2.shape)
# print(y2.shape)

# # x2 = x2[span*i: min(span*(i+1),len(x2))]
# # y2 = y2[span*i: min(span*(i+1),len(y2))]


# df2 = get_interaction_df(x2, y2, metadata2)
# df2.to_pickle(dataFolder2+fileName+".pkl")


# # sampled_df2 = df2.sample(frac=0.1, random_state=42)  # Set random_state for reproducibility
# # sampled_df2.to_pickle(dataFolder2+"data/"+fileName+'_s'+str(i)+"_10pct.pkl")
# print("done for Encode4")
# del x2, y2

# # 3. combine Encode3 and Encode4
# df = pd.concat([df1, df2])
# print(len(df))
# df.to_pickle("/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/"+fileName+".pkl")
# # df.to_pickle("/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/"+fileName+"_s"+str(i)+".pkl")




# sampled_df = pd.concat([sampled_df1, sampled_df2])

# sorted_df = df.sort_values(by=df.columns[0])
# print(sorted_df.head())
# print(sorted_df.tail())
# print(len(sorted_df))

# # # 4. downsampling the large df to 10% data
# # sampled_df = sorted_df.sample(frac=0.1, random_state=42)  # Set random_state for reproducibility
# sampled_df.to_pickle("/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/"+fileName+"_1pct.pkl")


# # 5. write the whole df to file
# sorted_df.to_pickle("/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/"+fileName+".pkl")
# # df.to_pickle("/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/"+"data/"+cellType+"_"+fileName+".pkl")
# # df.to_pickle(dataFolder+"data/"+cellType+"_"+fileName+"_balanced.pkl")
# # df.to_pickle(dataFolder+"data/"+cellType+"_"+fileName+"_true_nega.pkl")




# # subset to specific TF
# protein = "CTCF"
# df = df[df.protein==protein]
# print(len(df))
# df.to_pickle(dataFolder+"data/"+cellType+"_"+protein+"_"+fileName+".pkl")
