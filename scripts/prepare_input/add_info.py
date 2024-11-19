import pandas as pd
import sys

input_prefix=sys.argv[1]
inputFolder=sys.argv[2]
outputFolder=sys.argv[3]


# read two fasta files and add labels 
f1 = outputFolder+input_prefix+'.fasta'
f2 = outputFolder+input_prefix+'_shuffled.fasta'
df = pd.read_csv(f1, header=None)
new_df = pd.DataFrame({'ID':df[0].iloc[::2].values, 'seq':df[0].iloc[1::2].values, 'label':1})
df2 = pd.read_csv(f2, header=None)
new_df2 = pd.DataFrame({'ID':df2[0].iloc[::2].values, 'seq':df2[0].iloc[1::2].values, 'label':0})

# combine two dataframes
o = pd.concat([new_df, new_df2]).sort_values('ID')

# add protein info
file = '/new-stg/home/cong/DPI/downloads/Encode4_metadata.tsv'
df = pd.read_csv(file, sep='\t')
o['TF'] = df[df['File accession']==input_prefix]['Experiment target'].values[0].split('-')[0]

# write to csv file
o.to_csv(outputFolder+input_prefix+'.csv', header=None, index=None, sep='\t')