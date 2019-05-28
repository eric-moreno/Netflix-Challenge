# Loads training data into base.npy 

import numpy as np
import pandas as pd
import math

''' 0. Initialize global variables'''
N_MOVIES = 17770
N_USERS = 480189

''' 1. Load base and qual data '''
def load_base_qual():
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
	base = []
	qual = []

	for i in range(len(idx)):
		if idk[i][0] == 1:
			base.append(df[i])
		if idk[i][0] == 5:
			qual.append(df[i])

	# write to .txt files for c++ usage
	base = open("base.dta", "w")
	qual = open("qual.dta", "w")
	for i in range(len(idx)):
		if idk[i][0] == 1:
			base.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) +" " + str(df[i][3]) + "\n");
		if idk[i][0] == 5:
			qual.write(str(df[i][0]) + " " + str(df[i][1]) + "\n")
        
	base.close()
	qual.close()

# Creates predictions.npy by BellKor Baseline method
# Produces...
#	b_i.dta 	- movie bias
#	b_u.dta		- user bias
#	t_i.dta		- movie times
#	t_u.dta 	- user times

def generate_movie_data():
	print ("Generating movie data... \n")
	print ("Loading base data... \n")

	base = pd.read_table('base.dta', delim_whitespace=True,header=None)
	base = np.array(base)

	t_i = open("t_i.dta", "w")
	b_i = open("b_i.dta", "w")
	
	ratings = [[] for _ in range(N_MOVIES)]		
	
	print ("Generating movie time data... \n")
	# Read training data
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3] 
		ratings[m].append(r)
		t_i.write(str(t) + "\n")
		
	t_i.close()

	print ("Computing regularized movie averages... \n")
	# Compute movie averages
	averages = []
	movie_avg = 0
	total = 0
	for r in ratings: # compute average rating
		if (r == []):
			averages.append(0.0)
		else:
			avg = np.average(np.array(r)) 
			averages.append(avg)

	# Compute standard deviation, b_i = Sum(|r - u|) / (lambda1 - |R|)
	i = 0
	for r in ratings: 	
		if (r == []):
			movie_matrix.append([0.0, 0.0])
		else:
			r = np.array(r) # lambda1 = 25 (bellkor paper)
			sum = 0.0
			for x in r:
				sum = sum + (x - movie_avg)
			avg = sum/(25 + np.size(r))
			b_i.write(str(avg) + "\n")
			i = i + 1

	b_i.close()
	print("Finished generating movie data.\n")

def generate_user_data():
	print ("Generating user data... \n")
	print ("Loading base data... \n")

	base = pd.read_table('base.dta', delim_whitespace=True,header=None)
	base = np.array(base)
	
	t_u = open("t_u.dta", "w")
	b_u = open("b_u.dta", "w")
	
	# (r - u - b_i) values per user, sorted by user index
	user_ratings = [[] for _ in range(N_USERS+1)] 
	
	print ("Generating movie time data... \n")
	# Read training data 
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3]  
		# store ratings and movies by user index
		user_ratings[u].append(r - movie_avg - movie_matrix[m][1])
		t_u.write(str(t) + "\n")

	t_u.close()

	print ("Computing regularized user averages... \n")
	# Compute regularized user deviations: b_u = sum(r - u - b_i) / (lambda2 - R)
	# lambda2 = 10 (bellkor paper)
	for user in user_ratings:
		R = np.size(np.array(user))	 # total number of ratings
		sum = np.sum(np.array(user)) # sum of user's (r - u - b_1) 
		b = sum / (10 - R)
		b_u.write(str(b) + "\n")

	user_matrix = np.array(b_u)
	np.save('b_u.npy', user_matrix)

	print("Finished user computations.\n")

generate_movie_data()
generate_user_data()
'''
	user = 0
	error = []
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3] 
		prediction = movie_avg + movie_matrix[user][1]  + b_u[user]
		error.append(abs(prediction - r)*abs(prediction - r))
		user += 1

	rmse = math.sqrt(np.mean(error))
	print("RMSE: " + str(rmse))
'''