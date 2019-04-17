# Loads training data into base.npy 

import numpy as np
import pandas as pd

''' 1. Load data '''
df = pd.read_table('all.dta', delim_whitespace=True,header=None)
df = np.array(df)

idx = pd.read_table('all.idx', delim_whitespace=True,header=None)
idk = np.array(idx)

# Initialize dataframe 
for i in df: 
    i[0] -= 1
    i[1] -= 1
    
print("Finished loading data.\n")

''' 2. Get indices for training and testing data '''

# write to .txt files for c++ usage
base = open("base.txt", "w")
qual = open("qual.txt", "w")
for i in range(len(idx)):
    if idk[i][0] == 1:
        base.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) +" " + str(df[i][2]));
    if idk[i][0] == 5:
        qual.write(str(df[0]) + " " + str(df[i][1]))

base.close()
qual.close()