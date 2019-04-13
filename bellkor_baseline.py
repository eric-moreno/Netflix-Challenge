# Given full matrix of data of format: 
# (user_index, movie_index, rating)
# Computes the baseline predictions for the unknown movies using the BellKor formula:
# prediction = rating + dev_user + dev_movie

import numpy as np

# 0. Initialize filenames
data_filename = "full_matrix.txt"
pred_filename = "predictions.npy"

# 1. Load movie data matrix (averages and std_deviations)
avg_dev = np.load('avg_dev.npy')
print(avg_dev)

u = avg_dev[0][0]  # Get overall movie average
avg_dev = np.delete(avg_dev, 0, 0) # remove top row

# 2. Compute user average rating deviations -> user_devs
user_ratings = [] 	# 2d array of (r - u - b_1) per user
user_movs = []  # movies rated by user (rows: user_idx, cols: movie_idx's)
user = 0		# index of current user in file

# Fill in user_devs and user_movs
data_file = open(data_filename, "r")
for line in data_file:
	vals = [int(x) for x in line.split()]		# get list of all ratings
	u, m, r = vals[0], vals[1], vals[2]			# user, movie, rating

	if (user < u+1):
		user_ratings.append([(r - u - avg_dev[m][1])])
		user_movs.append([m])
		user = user + 1
	else:
		user_ratings[u].append(r - u - avg_dev[m][1])
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
		
# 4. Predict unknown ratings -> "predictions.npy"

data_file = open(data_filename, "r")
predictions = [] # matrix: user_index, movie_index, predicted_rating

usr_idx = -1
for movs in user_movs:
	usr_idx = usr_idx + 1
	mov_idx = -1 
	for [avg, b_i] in avg_dev:
		mov_idx = mov_idx + 1
		if mov_idx not in movs:
			rating = u + b_u[usr_idx] + b_i
			predictions.append([usr_idx, mov_idx, rating])

print(predictions)
# save predictions matrix
np.save("predictions.npy", predictions)
# close all files
data_file.close()