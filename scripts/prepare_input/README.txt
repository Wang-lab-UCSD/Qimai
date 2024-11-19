######job scripts##########
1. process data as DeepBind---------------------------
1.1 get fasta
> sub_get_fasta.job processes the downloaded narrow peak files to four-column csv file, which is "ID", "DNA sequence", "label" and "TF"

> sub_get_fasta_v2.job did the similar job as original script with only difference which is to extract a fixed-length sequence centered at the point source for each called peak (peak summit, the 10th column in peak file)

> sub_get_fasta_v3.job did the similar job as original. If the sequence is longer than 512bp, extract a 512-bp sequence centered at the peak summit otherwise keep as it is.

> sub_get_fasta_v4.job did the similar job as v3. The only difference: the label is peak intensity, calculated as RPM, instead of binary. It calls add_labels_v2 function. The above scripts call add_labels function

1.2 aggegate and downsampling



2. process data as DeepSEA-----------------------------
2.1 get data matrix
> sub_build_deepsea_dataset.job builds deepsea training dataset. See https://github.com/jakublipinski/build-deepsea-training-dataset for details. It requires the metadata and the bed folder. The script will generate 9 files in the out folder.

>> (train|valid|test).mat files contain both data (DNA sequences) and labels for training, validation and testing respectively. The format of the files is same as the ones provided by the paper authors.

>> (train|valid|test)_(data|labels).npy files contain data and labels for training, validation and testing saved in the .npy format.

2.2 get interaction pair
> sub_get_interaction_pair.job reorganizes the '*.mat' files into the format required by our transformer model

3. build customized data------------------------------
3.1 get protein embeddings from localColabFold
check folder $HOME/DPI/colabfold/output/ for details

3.2 prepare tested DNA sequences

3.3 build protein-DNA pair for testing
First, prepare file with at least two columns: 'dna' and 'protein'. 'dna' column is dna sequence, like 'AGAGGCTTGCTTTT', protein column is protein name like 'GATA3' or 'BRCA1'

Then run script generate_dataset_with_embedding.py under this folder

3.4 Some modifications
> generate_database_for_dna_and_protein_embedding.py: generates pickle file to store embeddings for DNA sequences and proteins separately

> generate_DNA_lmdb_file.py: pickle file is not efficient for large datasets. Considering the relative simple data structure, we chose LMDB file instead of HDF5 file for extremely fast read operations. It uses DNABERT2 to extract DNA embeddings

###### datasets ##########

> 690protein_name_to_ENSG.csv is mapping between protein names and Ensembl id (ENSGxx). It's downloaded on 02.14.2023 from https://biit.cs.ut.ee/gprofiler/convert with the proteins_encode3_and_Chip690.txt as input. 

> TF_family.xlsx is subset of TF_annotation.xlsx TableS1, which is downloaded from Lambert, Samuel A., et al. "The human transcription factors." Cell 172.4 (2018): 650-665.

> proteins.txt is 619 proteins from Encode3 dataset while proteins_encode3_and_Chip690.txt is 690 proteins from both Encode3 and ChIP690 datasets

