# Loads training data into base.npy

import numpy as np
import pandas as pd

''' 1. Load data '''
df = pd.read_table('E:\\Documents\\Caltech\\CS156\\mu\\all.dta', delim_whitespace=True, header=None)
df = np.array(df)

idx = pd.read_table('E:\\Documents\\Caltech\\CS156\\mu\\all.idx', delim_whitespace=True, header=None)
idk = np.array(idx)

# Increment movie/user down by 1 if you need
#for i in df:
#    i[0] -= 1
#    i[1] -= 1

print("Finished loading data.\n")

''' 2. Get indices for training and testing data '''

# write to .txt files for c++ usage
base = open("E:\\Documents\\Caltech\\CS156\\base.dta", "w")
qual = open("E:\\Documents\\Caltech\\CS156\\qual.dta", "w")
for i in range(len(idx)):
    if idk[i][0] == 1:
        base.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) + " " + str(df[i][3]) + "\n")
    if idk[i][0] == 5:
        qual.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) + "\n")

base.close()
qual.close()