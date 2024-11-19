import random
import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(1)
dataFolder = '/new-stg/home/cong/DPI/dataset/Encode4/get_fasta_results_max_512bp/'
outputFolder = '/new-stg/home/cong/DPI/dataset/Encode4/csv_files/'
Path(outputFolder).mkdir(parents=True, exist_ok=True)
files = glob.glob(dataFolder+'*csv')
print(len(files))
print(files[0])

def read_plus(file):
    df = pd.read_csv(file,sep="\t",header=None,names=['loc','label','dna','protein'],dtype={'loc':str,'label':int,'dna':str,'protein':str})
    return df.sample(n=min(100,df.shape[0]), random_state=1)

df = pd.concat(list(map(read_plus, files)))
print(df.shape[0])

df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
df_train.to_csv(outputFolder+"100_per_exp_max_512bp_train.csv", sep = '\t', index=0, header=False)
df_val.to_csv(outputFolder+"100_per_exp_max_512bp_test.csv", sep = '\t', index=0, header=False)
