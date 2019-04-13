# Creates avg_dev.npy 
# 	- Computes the average rating and deviation of each movie in full_matrix.txt

import numpy as np

# Read full data matrix
data_filename = "full_matrix.txt"
data_file = open(data_filename, "r")

# Get all ratings of each movie
ratings = [] 	
len = 0
for line in data_file:
	vals = [int(x) for x in line.split()]		
	u, m, r = vals[0], vals[1], vals[2]			# user, movie, rating
	if (len < m+1):
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

for r in ratings: 	# compute standard deviation, b_i = Sum(r - u) / (lambda1 - |R|)
	r = np.array(r) # lambda1 = 25 (bellkor paper)
	sum = 0.0
	for x in r:
		sum = sum + (x - movie_avg)
	dev = sum/(25 + np.size(r))
	matrix.append([avg, dev])

np_matrix = np.array(matrix)
np.save('avg_dev.npy', np_matrix) # Store matrix in avg_dev.npy

# Close data file
data_file.close()