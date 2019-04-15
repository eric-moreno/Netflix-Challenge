# Creates avg_dev.npy 
# 	- Computes the average rating and deviation of each movie in full_matrix.txt

import numpy as np
import pandas as pd

# load data
df = pd.read_table('all.dta', delim_whitespace=True,header=None)
df = np.array(df)

idx = pd.read_table('all.idx', delim_whitespace=True,header=None)
idk = np.array(idx)

# initialize dataframe 
for i in df: 
    i[0] -= 1
    i[1] -= 1

# get indices for training and validation
base = []
valid = []

for i in range(len(idx)):
    if idk[i][0] == 1:
        base.append(i)
    if idk[i][0] == 2:
        valid.append(i)

# read training data from all.dta
ratings = []
len = 0
for i in base:
	u, m, t, r =  df[i][0], df[i][1], df[i][2], df[i][3]
	if (len < m + 1):
		ratings.append([r])
		len += 1
	else:
		ratings[m].append(r)
	
# Compute averages and regularized deviations 
averages = []
for r in ratings: # compute average rating
	r = np.array(r)		
	avg = np.average(r) 
	averages.append(avg)

movie_avg = np.average(np.array(averages)) 

# Construct Matrix
# 	- First row contains overall movie average rating
# 	- Consequent Row Format: [average, deviation]
matrix = [[movie_avg, 0.0]]	
i = 0 # index of movie
for r in ratings: 	# compute standard deviation, b_i = Sum(r - u) / (lambda1 - |R|)
	r = np.array(r) # lambda1 = 25 (bellkor paper)
	sum = 0.0
	for x in r:
		sum = sum + (x - movie_avg)
	dev = sum/(25 + np.size(r))
	matrix.append([averages[i], dev])
	i = i + 1

np_matrix = np.array(matrix)
np.save('avg_dev.npy', np_matrix) # Store matrix in avg_dev.npy
