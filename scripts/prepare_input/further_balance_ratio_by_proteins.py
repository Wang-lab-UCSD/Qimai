import h5py
import numpy as np
import pandas as pd
import sys
import random
from sklearn.utils import resample


dataset = sys.argv[1] 
file = sys.argv[2] 
print(dataset)
print(file)

dataFolder = "/new-stg/home/cong/DPI/dataset/"+dataset+"/deepsea/data/"

    
df = pd.read_pickle(dataFolder + file + '.pkl')
print(df.shape[0])
print(df.head())

# Grouping the dataframe by 'protein'
grouped = df.groupby('protein')

# Create an empty dataframe to store the balanced data
balanced_df = pd.DataFrame()

# Iterate over groups, resample to balance classes, and append to balanced_df
for group_name, group_df in grouped:
    # Separate majority and minority classes
    class_0 = group_df[group_df['label'] == 0]
    class_1 = group_df[group_df['label'] == 1]
    if len(class_0) <= len(class_1):
        minority_class = class_0
        majority_class = class_1
    else:
        minority_class = class_1
        majority_class = class_0
    if len(minority_class)>=1:
        # Resample the minority class to match the length of the majority class
        minority_class_resampled = resample(minority_class,
                                        replace=True,  # Set to True for oversampling
                                        n_samples=len(majority_class),
                                        random_state=42)
    
    # Combine resampled minority class with majority class
    balanced_group = pd.concat([majority_class, minority_class_resampled])
      
    # Append the balanced data to the final dataframe
    balanced_df = pd.concat([balanced_df, balanced_group])


# Display the balanced dataframe
print(balanced_df)
print(balanced_df.shape[0])
balanced_df.to_pickle(dataFolder+file+'_ratio_balanced_by_protein.pkl')  
examples_per_protein = int(np.mean(balanced_df.protein.value_counts()))
print('number of examples per protein: '+str(examples_per_protein))
print("success!")

# # sample the balanced df to make sure each protein has the same number of examples
# balanced_group2 = resample(balanced_group,replace=True,n_samples=examples_per_protein)
# balanced_df.to_pickle(dataFolder+file+'_full_balance_by_protein.pkl')   # ratio and total number are both balanced