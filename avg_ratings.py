# Computes the average rating of each movie in full_matrix.txt

import numpy as np

data_filename = "full_matrix.txt"
avg_filename = "averages.txt"

# Read full data matrix
data_file = open(data_filename, "r")

ratings = [] 	# 2d array of ratings of movies, sorted by index
len = 0

for line in data_file:
	vals = [int(x) for x in line.split()]		# get list of all ratings
	u, m, r = vals[0], vals[1], vals[2]			# user, movie, rating
	if (len < m+1):
		ratings.append([r])
		len += 1
	else:
		ratings[m].append(r)

averages = list(map(lambda arr: np.average(np.array(arr)), ratings))

# Write averages to avg_filename file
avg_file = open(avg_filename, "w")
for avg in averages:
	avg_file.write("%.4f\n" % avg)

# Close open files
data_file.close()
avg_file.close()