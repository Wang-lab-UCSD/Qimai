import numpy as np
import pandas as pd


dataFolder = '/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/data/'

def main(tag):
    
    dpi = pd.read_pickle(dataFolder+'all_'+tag+'_wide.pkl')
    print(dpi.shape)
    print(dpi.head())
    
    dpi_l = pd.melt(dpi.reset_index(), id_vars = 'dna', var_name='protein', value_name='label')
    print(dpi_l.shape)
    print(dpi_l.head())
    print(dpi_l.label.value_counts())
    dpi_l.to_pickle(dataFolder+'all_'+tag+'_long.pkl')
    
    
    
main('train')
main('valid')
main('test')
