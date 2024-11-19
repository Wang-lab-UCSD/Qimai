import numpy as np
import pandas as pd

def dna_to_one_hot(dna_sequence):
    one_hot_encoding = []
    for nucleotide in dna_sequence:
        if nucleotide == 'A':
            one_hot_encoding.append([1, 0, 0, 0])
        elif nucleotide == 'G':
            one_hot_encoding.append([0, 1, 0, 0])
        elif nucleotide == 'C':
            one_hot_encoding.append([0, 0, 1, 0])
        elif nucleotide == 'T':
            one_hot_encoding.append([0, 0, 0, 1])
        else:
            raise ValueError("Invalid nucleotide in the DNA sequence: " + nucleotide)
    
    return np.asarray(one_hot_encoding, dtype=np.uint8)

# dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/data/'
dataFolder = '/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/'

def main(tag):

    dpi = pd.read_pickle(dataFolder+tag+'.pkl')
    df = pd.pivot_table(dpi, index='dna', columns='protein',values='label', fill_value=0)
    y = df.to_numpy(dtype=np.uint8)
    print(y.shape)

    dnas = df.index
    print(dnas.shape)
    x = np.asarray([dna_to_one_hot(i) for i in dnas])
    print(x.shape)

    np.save(dataFolder+tag+'_x.npy', x)
    np.save(dataFolder+tag+'_y.npy', y)
    
    
# main('train')
# main('valid')
# main('test')

main('train_min10')
main('valid_min10')
main('test_min10')
