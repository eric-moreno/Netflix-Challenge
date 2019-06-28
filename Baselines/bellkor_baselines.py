'''
 Baselines1
 BellKor Baseline Method
 
	load_base_qual
	base.dta
	qual.dta
	
	generate_movie_data
	t_i.dta		- movie times (0 indexed)
 	b_i.dta 	- regularized movie averages (0 indexed)
 	
 	generate_user_data
 	t_u.dta 	- user times (0 indexed)
	b_u.dta		- regularized user bias (0 indexed)
	ratings.dta - user ratings (movie, rating, time) (1 indexed)
'''

import numpy as np
import pandas as pd
import math

''' 0. Initialize global variables'''
N_MOVIES = 17770
N_USERS = 480189
AVG_RATING = 3.6033;

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
	probe = []
	qual = []

	for i in range(len(idx)):
		if idk[i][0] == 1:
			base.append(df[i])
		if idk[i][0] == 4:
			probe.append(df[i])
		if idk[i][0] == 5:
			qual.append(df[i])

	# write to .txt files for c++ usage
	base = open("base.dta", "w")
	probe = open("probe.dta", "w")
	qual = open("qual.dta", "w")
	for i in range(len(idx)):
		if idk[i][0] == 1:
			base.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) +" " + str(df[i][3]) + "\n");
		if idk[i][0] == 4:
			probe.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) +" " + str(df[i][3]) + "\n");
		if idk[i][0] == 5:
			qual.write(str(df[i][0]) + " " + str(df[i][1]) + " " + str(df[i][2]) +" " + str(df[i][3]) + "\n");
        
	base.close()
	probe.close()
	qual.close()

def generate_movie_data():
	print ("Generating movie data... \n")
	print ("Loading base data... \n")

	base = pd.read_table('base.dta', delim_whitespace=True,header=None)
	base = np.array(base)

	t_i = open("t_i.dta", "w")
	b_i = open("b_i.dta", "w")
	
	ratings = [[] for _ in range(N_MOVIES+1)]

	print ("Generating movie time data... \n")
	# Read training data
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3] 
		ratings[m].append(r)
		t_i.write(str(t) + "\n")
		
	t_i.close()

	print ("Computing regularized movie averages... \n")
	'''
	# Compute movie averages
	averages = []
	total = 0
	for r in ratings: # compute average rating
		if (r == []):
			averages.append(0.0)
			b.i.write(str(0.0) + "\n")
		else:
			avg = np.average(np.array(r)) 
			averages.append(avg)
			b.i.write(str(avg - AVG_RATING) + "\n")
	'''
	# Compute standard deviation, b_i = Sum(|r - u|) / (lambda1 + |R|)
	i = 0
	for r in ratings: 	
		if (r == []):
			b_i.write("0.0\n")
		else:
			r = np.array(r) # lambda1 = 25 (bellkor paper)
			sum = 0.0
			for x in r:
				sum = sum + (x - AVG_RATING)
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
	
	b_i = pd.read_table('b_i.dta', delim_whitespace=True,header=None)
	b_i = np.array(b_i)
	
	t_u = open("t_u.dta", "w")
	b_u = open("b_u.dta", "w")
	ratings = open("ratings.dta", "w")
	
	# (r - u - b_i) values per user, sorted by user index
	nums = [[] for _ in range(N_USERS+1)] 
	
	# array of [movie, rating] sorted by user index
	user_ratings = [[[]] for _ in range(N_USERS+1)] 
	
	# Read training data 
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3]  
		# store ratings and movies by user index
		dev_i = b_i[m] #- AVG_RATING
		nums[u].append(r - AVG_RATING - dev_i)
		user_ratings[u].append([m, r, t])
		t_u.write(str(t) + "\n")
	
	ratings.write(str(user_ratings))
	ratings.close()
	
	t_u.close()

	print ("Computing regularized user averages... \n")
	# Compute regularized user deviations: b_u = sum(r - u - b_i) / (lambda2 + R)
	# lambda2 = 10 (bellkor paper)
	for user in nums:
		R = np.size(np.array(user))	 # total number of ratings
		sum = np.sum(np.array(user)) # sum of user's (r - u - b_1) 
		b = sum / (10 + R)
		b_u.write(str(b) + "\n")

	b_u.close()

	print("Finished user computations.\n")

def generate_baselines():
	print("Generate Baseline Predictions:\n")
	
	print("Loading movie bias data...\n")
	
	b_i = pd.read_table('b_i.dta', delim_whitespace=True,header=None)
	b_i = np.array(b_i)
	
	print("Loading user bias data...\n")
	
	b_u = pd.read_table('b_u.dta', delim_whitespace=True,header=None)
	b_u = np.array(b_u)
	
	print ("Loading qual data... \n")

	qual = pd.read_table('base.dta', delim_whitespace=True,header=None)
	qual = np.array(qual)
	
	print("Computing bellkor baseline predictions...\n")
	baselines = open("baselines1.dta", "w")
	
	user = 0
	error = []
	for i in qual:
		# user, movie, time, rating
		u, m, r =  i[0], i[1], i[3] 
		
		if (u < 17771): 
			prediction = AVG_RATING + b_u[u] + b_i[i]		
			baselines.write(str(prediction[0][0]) + "\n")
			error.append(abs(prediction[0][0] - r) * abs(prediction[0][0] - r))
		
			user += 1
			if (user % 100000 == 0):
				print(str(user) + " data points, RMSE: " + str(math.sqrt(np.mean(error))) + "")
			
	rmse = math.sqrt(np.mean(error))
	print("RMSE: " + str(rmse))

	baselines.close()
	
	print("Finished generating baseline predictions.")


load_base_qual()
#generate_movie_data()
#generate_user_data()
#generate_baselines()
