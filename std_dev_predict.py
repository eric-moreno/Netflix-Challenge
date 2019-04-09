# Predict unknown movie ratings using method of standard deviations
# predicted_rating = (avg_rating) + (user_std_dev)

import numpy as np

# 1. Read in average movie data -> averages

avg_file = open("averages.txt", "r")
averages = []
for avg in avg_file:
	averages.append(float(avg))

# Store std_dev's of users
std_devs = []

# 2. Compute user standard deviations -> std_devs

# Read full data matrix
data_file = open("full_matrix.txt", "r")
for line in data_file:
	ratings = [float(x) for x in line.split()]	# get list of all ratings
	devs = [] 									# deviations from average
	
	idx = 0										# index of movie
	for rating in ratings:
		if rating != 0:
			devs.append(rating - averages[idx])
		idx = idx + 1
			
	arr = np.array(devs)
	std_devs.append(np.average(arr))			# store standard deviation for user


# 3. Predict unknown ratings -> "predictions.txt"

data_file = open("full_matrix.txt", "r")
predictions = open("predictions.txt", "w")
user = 0										# index of user
for line in data_file:
	ratings = [float(x) for x in line.split()]	

	idx = 0										# index of movie
	for rating in ratings:
		if rating != 0:
			predictions.write("%.2f " % rating)
		else:
			prediction = averages[idx] + std_devs[user]
			# Ensure ratings are between 0 and 5
			if prediction > 5 :
				prediction = 5.00;
			if prediction < 0:
				prediction = 0.00
			predictions.write("%.2f " % prediction)
		idx = idx + 1
		
	predictions.write("\n")						# write to file
	user = user + 1
	
# close all files
data_file.close()
avg_file.close()
predictions.close()