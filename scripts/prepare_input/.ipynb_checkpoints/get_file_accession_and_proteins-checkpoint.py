import pandas as pd

file = '/new-stg/home/cong/DPI/downloads/Encode4_metadata.tsv'
df = pd.read_csv(file, sep='\t')
print('num of experiments: '+str(len(df)))
df['File accession'].to_csv('/new-stg/home/cong/DPI/dataset/Encode4/file_accession.txt', header=None, index=None)
df['Experiment target'].str.split('-', expand=True)[0].drop_duplicates().to_csv('/new-stg/home/cong/DPI/dataset/Encode4/proteins.txt', header=None, index=None)
print('num of proteins: '+str(len(df['Experiment target'].str.split('-', expand=True)[0].unique())))