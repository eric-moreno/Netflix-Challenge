# Given full matrix of data of format: 
# (user_index, movie_index, rating)
# Computes the baseline predictions for the unknown movies using the BellKor formula:
# prediction = rating + dev_user + dev_movie

import numpy as np

# 0. Initialize filenames
averages_filename = "averages.txt"
data_filename = "full_matrix.txt"
pred_filename = "predictions.txt"

# 1. Read in average movie data -> averages

avg_file = open(averages_filename, "r")
averages = []
for avg in avg_file:
	averages.append(float(avg))

avg_rating = np.mean(averages)

# 2. Compute each movie's standard deviation from average rating
movie_devs = []
for avg in averages:
	movie_devs.append(avg - avg_rating)

# 3. Compute user average ratings -> user_devs

user_devs = [] 	# 2d array of ratings of users, sorted by user index
user_movs = []  # 2d array of movies rated by user (rows: user_idx, cols: movie_idx's)
user = 0		# index of current user in file
# Read full data matrix
data_file = open(data_filename, "r")
for line in data_file:
	vals = [int(x) for x in line.split()]		# get list of all ratings
	u, m, r = vals[0], vals[1], vals[2]			# user, movie, rating

	if (user < u+1):
		user_devs.append([(r - averages[m])])
		user_movs.append([m])
		user = user + 1
	else:
		user_devs[u].append(r - averages[m])
		user_movs[u].append(m)

user_devs = list(map(lambda arr: np.average(np.array(arr)), user_devs))

# 4. Predict unknown ratings -> "predictions.txt"

data_file = open(data_filename, "r")
predictions = open(pred_filename, "w")

usr_idx = -1
for user in user_movs:
	usr_idx = usr_idx + 1
	mov_idx = -1
	for avg in averages:
		mov_idx = mov_idx + 1
		
		if mov_idx not in user:
			pred = avg_rating + user_devs[usr_idx] + movie_devs[mov_idx]
			pred = min(5, pred)	# cap at 5
			pred = max(0, pred) # cap at 0
			predictions.write("%d %d %.2f \n" % (usr_idx, mov_idx, pred))
	

# close all files
data_file.close()
avg_file.close()
predictions.close()