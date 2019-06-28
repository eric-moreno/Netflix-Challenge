# Creates predictions.npy by BellKor Baseline method

import numpy as np
import pandas as pd

''' 1. Load data '''
base = np.load('base.npy')
        
''' 2. Construct matrix of total average (u) and deviations (b_i) for each movie '''
ratings = []		# arrays of all movie ratings sorted by movie index
len = 0				# size of ratings

# Read training data
for i in base:
	# user, movie, time, rating
	u, m, t, r =  i[0], i[1], i[2], i[3] 

	if (len < m + 1):
		ratings.append([r])
		len += 1
	else:
		ratings[m].append(r)

# Compute movie averages and regularized deviations 
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

print("Finished movie computations.\n")

''' 3. Compute regularized user rating deviations (b_u). '''
user_ratings = [] 	# (r - u - b_1) values per user, sorted by user index
user_movs = []  	# arrays of movie indices rated by user, sorted by user index
user = 0			# index of current user in file

# Read training data 
for i in base:
	# user, movie, time, rating
	u, m, t, r =  i[0], i[1], i[2], i[3]  

	# store ratings and movies by user index
	if (user < u+1):
		user_ratings.append([(r - u - matrix[m][1])])
		user_movs.append([m])
		user = user + 1
	else:
		user_ratings[u].append(r - u - matrix[m][1])
		user_movs[u].append(m)
		
# Compute regularized user deviations: b_u = sum(r - u - b_i) / (lambda2 - R)
# lambda2 = 10 (bellkor paper)
b_u = []
for user in user_ratings:
	arr = np.array(user)
	R = np.size(arr)	# total number of ratings
	sum = np.sum(arr)  	# sum of user's (r - u - b_1) 
	b = sum / (10 - R)
	b_u.append(b)
		
print("Finished user computations.\n")

''' 4. Predict and store unknown ratings '''
# "predictions.npy"
predictions = [] # matrix: user_index, movie_index, predicted_rating

usr_idx = -1
for movs in user_movs:
	usr_idx = usr_idx + 1
	mov_idx = -1 
	for [avg, b_i] in matrix:
		mov_idx = mov_idx + 1
		if mov_idx not in movs:
			rating = u + b_u[usr_idx] + b_i
			predictions.append([usr_idx, mov_idx, rating])

print(predictions)
# save predictions matrix
np.save("predictions.npy", predictions)
print("Finished predictions.\n")
print("All done!")
