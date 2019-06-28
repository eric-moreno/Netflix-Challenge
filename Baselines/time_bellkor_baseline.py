# Creates predictions.npy by BellKor Baseline method with time-based binning
# Produces...
#	b_it.npy	- movie-time deviations
#	b_ut.npy	- user-time deviations

import numpy as np
import pandas as pd
import math

''' 0. Initialize static and global variables '''
N_MOVIES = 17770
AVG_RATING = 3.6033;
N_USERS = 480189
N_BINS = 30

def generate_movie_time_data():
	b_i = pd.read_table('b_i.dta', delim_whitespace=True,header=None)
	b_i = np.array(b_i)
	
	t_i = pd.read_table('t_i.dta', delim_whitespace=True,header=None)
	t_i = np.array(t_i)
	
	print("Finished loading data...\n")

	print ("Computing time-dependent movie data... \n")

	bins = np.linspace(0, max(t_i), N_BINS)
	bin_idx = np.digitize(t_i, bins.reshape(len(bins)))
	
	print(bin_idx)
	
	avg_ratings = [[] for _ in range(N_BINS+1)] 
	# store each b_i into appropriate bin
	i = 0
	for item in b_i:
		bin = bin_idx[i][0]
		avg_ratings[bin].append(item[0])
		i += 1
	
	b_it = []
	for row in avg_ratings:
		if (row == []):
			b_it.append([])
		else:
			b_it.append(np.mean(row))

	print(b_it)
	file = open("b_it.dta", "w")
	file.write(str(b_it))	
	file.close()
	print("Finished computing time-dependent movie data... \n")
	

def compute_user_time_averages():
	print ("Generating time-dependent user data... \n")
	
	print ("Loading base data... \n")

	base = pd.read_table('base.dta', delim_whitespace=True,header=None)
	base = np.array(base)

	print ("Computing user time averages... \n")
	times = [[] for _ in range(N_USERS+1)]
	
	# Read training data
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3] 
		times[u].append(t)
		
	file = open("t_avg.dta", "w")
	
	avg_times = [_ for _ in range(N_USERS+1)]
	idx = 1
	for row in times:
		avg_times[idx] = np.mean(row)
		index += 1
	
	file.write(str(avg_times))
	file.close()

def generate_baselines():
	print("Generate Baseline Predictions:\n")
	
	print("Loading movie bias data...\n")
	
	b_it = pd.read_table('b_i.dta', delim_whitespace=True,header=None)
	b_it = np.array(b_i)
	
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
			prediction = AVG_RATING + b_u[u] + b_it[i]		
			baselines.write(str(prediction[0][0]) + "\n")
			error.append(abs(prediction[0][0] - r) * abs(prediction[0][0] - r))
		
			user += 1
			if (user % 100000 == 0):
				print(str(user) + " data points, RMSE: " + str(math.sqrt(np.mean(error))) + "")
			
	rmse = math.sqrt(np.mean(error))
	print("RMSE: " + str(rmse))

	baselines.close()
	
	print("Finished generating baseline predictions.")

generate_movie_time_data()
generate_baselines()












