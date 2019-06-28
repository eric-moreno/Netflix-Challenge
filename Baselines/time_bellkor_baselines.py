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
ALPHA = 0.01

def generate_movie_time_data():
	b_i = pd.read_table('b_i.dta', delim_whitespace=True,header=None)
	b_i = np.array(b_i)
	
	t_i = pd.read_table('t_i.dta', delim_whitespace=True,header=None)
	t_i = np.array(t_i)
	
	print("Finished loading data...\n")

	print ("Computing time-dependent movie data... \n")

	bins = np.linspace(0, max(t_i), N_BINS)
	bin_idx = np.digitize(t_i, bins.reshape(len(bins)))
	
	avg_ratings = [[] for _ in range(N_BINS+1)] 
	# store each b_i into appropriate bin
	i = 0
	for item in b_i:
		bin = bin_idx[i][0]
		avg_ratings[bin].append(item[0])
		i += 1
	
	avg_bias = []
	overall_bias = 0
	cnt = 0
	for row in avg_ratings:
		if (row == []):
			avg_bias.append(0)
		else:
			avg_bias.append(np.mean(row))
			overall_bias += np.mean(row)
			cnt += 1

	overall_bias = overall_bias/cnt

	# Compute deviation for each bin
	b_t = []
	for b in avg_bias:
		b_t.append(b - overall_bias)

	# Recompute user deviations with bin deviations
	# b_it = b_i + b_t(bin)
	file = open("b_it.dta", "w")
	i = 0
	for item in b_i:
		bin = bin_idx[i][0]
		file.write(str(item + b_t[bin]) + "\n")
		i += 1
		
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
	for row in times:
		file.write(str(np.mean(row)) + "\n")
	file.close()

def generate_user_time_data():
	print ("Generating time-dependent user data... \n")
	
	print ("Loading base data... \n")

	base = pd.read_table('base.dta', delim_whitespace=True,header=None)
	base = np.array(base)
	
	print ("Loading user data... \n")
		
	b_u = pd.read_table('b_u.dta', delim_whitespace=True,header=None)
	b_u = np.array(b_u)
	b_u.reshape(len(b_u))
	
	print ("Sorting ratings by user... \n")
	
	# separate users
	users = [[] for u in range(N_USERS)]
	times = [[] for u in range(N_USERS)]
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3]
		users[u-1].append([m, t, r])
		times[u-1].append(t) 
		
	file = open("t_avg.dta", "w")
	for t in times:
		if len(t) == 0:
			file.write("0.0 \n");
		else:
			file.write(str(np.mean(t)) + "\n");
	file.close()
	
	print ("Loading user time average data... \n")
	
	t_avg = pd.read_table('t_avg.dta', delim_whitespace=True,header=None)
	t_avg = np.array(t_avg)
	t_avg.reshape(len(t_avg))
	
	# For each user, store b and alpha
	b_ut = [[b_u[u][0], 0] for u in range(N_USERS)]
	
	print ("Running linear regression to compute user-time data... \n")
	
	# Linear regression by gradient descent for each user
	u = 0
	for ratings in users:
		b_grad = b_u[u][0];
		a_grad = 0;
		b, a = b_ut[u][0], b_ut[u][1] # current b and a for user
		
		N = float(len(ratings))
		for i in range(len(ratings)):
			m, t, r = ratings[i][0], ratings[i][1], ratings[i][2]
			# compute dev_u
			dev = abs(t - t_avg[u][0]) ** 0.4
			if (dev < 0):
				dev = (-1) * dev
			p = b + a * dev;
			
			b_grad += -(2/N) * (r - p)
			a_grad += -(2/N) * dev * (r - p)
		
		# update user parameters -- b and a
		b_ut[u][0] = b - (1 * b_grad)
		b_ut[u][1] = m - (1 * a_grad)
		if (u % 10000 == 0):
			print("Finished " + str(u) + " users.\n")
			
		u += 1
	
	file = open("b_ut.dta", "w")
	for user in b_ut:
		file.write(str(user[0])+" "+ str(user[1])+"\n")
	file.close()

	print("Finished computing time-dependent user data... \n")
		
def generate_baselines():
	print("Generate Baseline Predictions:\n")
	
	print("Loading movie bias data...\n")
	
	b_it = pd.read_table('b_it.dta', delim_whitespace=True,header=None)
	b_it = np.array(b_it)
	
	print("Loading user bias data...\n")
	
	b_ut = pd.read_table('b_ut.dta', delim_whitespace=True,header=None)
	b_ut = np.array(b_ut)
		
	b_u = pd.read_table('b_u.dta', delim_whitespace=True,header=None)
	b_u = np.array(b_u)
	b_u.reshape(len(b_u))
	
	print ("Loading qual data... \n")

	qual = pd.read_table('base.dta', delim_whitespace=True,header=None)
	qual = np.array(qual)
	
	print ("Loading user time average data... \n")
	
	t_avg = pd.read_table('t_avg.dta', delim_whitespace=True,header=None)
	t_avg = np.array(t_avg)
	t_avg.reshape(len(t_avg))
	
	print("Computing time-based bellkor baseline predictions...\n")
	baselines = open("baselines3.dta", "w")
	
	user = 0
	error = []
	for i in qual:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3] 
		
		if (u < 17771): 
			dev = abs(t - t_avg[u-1][0]) ** 0.4
			if (dev < 0):
				dev = (-1) * dev
				
			b = b_ut[u-1][0]
			a = b_ut[u-1][1]
			u_term = (b + a * dev);
			
			if (u_term > 5) or (u_term < -5):
				u_term = b_u[u][0]	# something went wrong
			
			i_term = float(b_it[m-1][0].replace('[','').replace(']',''))
			
			prediction = AVG_RATING + u_term + i_term

			if (prediction > 5):
				prediction = 5
			if (prediction < 0):
				prediction = 0
				
			#print(prediction)
			baselines.write(str(prediction) + "\n")
			error.append(abs(prediction - r) * abs(prediction - r))
			
			user += 1
			if (user % 100000 == 0):
				print(str(user) + " data points, RMSE: " + str(math.sqrt(np.mean(error))) + "")
			
	rmse = math.sqrt(np.mean(error))
	print("RMSE: " + str(rmse))

	baselines.close()
	
	print("Finished generating baseline predictions.")
	
#generate_movie_time_data()
#compute_user_time_averages()
generate_user_time_data()
generate_baselines()













