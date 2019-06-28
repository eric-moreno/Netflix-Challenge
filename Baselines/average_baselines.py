'''
 Baselines0
 Average Baseline Method
'''

import numpy as np
import pandas as pd
import math

''' 0. Initialize global variables'''
N_MOVIES = 17770
N_USERS = 480189
AVG_RATING = 3.6033;

def generate_baselines():
	print ("Loading base data... \n")

	base = pd.read_table('base.dta', delim_whitespace=True,header=None)
	base = np.array(base)

	mov_ratings = [[] for _ in range(N_MOVIES+1)]
	usr_ratings = [[] for _ in range(N_USERS+1)]

	# Read training data
	for i in base:
		# user, movie, time, rating
		u, m, t, r =  i[0], i[1], i[2], i[3] 
		mov_ratings[m].append(r)
		usr_ratings[u].append(r)

	print ("Computing movie averages... \n")
	
	mov_averages = [0]
	for m in mov_ratings: # compute average rating
		if (m == []):
			mov_averages.append(AVG_RATING)
		else:
			avg = np.average(np.array(m)) 
			mov_averages.append(avg)

	print ("Computing user averages... \n")
	
	usr_averages = [0]
	for u in usr_ratings: # compute average rating
		if (u == []):
			usr_averages.append(AVG_RATING)
		else:
			avg = np.average(np.array(u)) 
			usr_averages.append(avg)
			
	print ("Loading qual data... \n")

	qual = pd.read_table('base.dta', delim_whitespace=True,header=None)
	qual = np.array(qual)
	
	print("Computing average baseline predictions...\n")
	
	baselines = open("baselines0.dta", "w")
	
	user = 0
	error = []
	for i in qual:
		# user, movie, time, rating
		u, m, r =  i[0], i[1], i[3] 
		
		if (u < 17771): 	
			dev_u = usr_averages[u] - AVG_RATING
			dev_i = mov_averages[m] - AVG_RATING
			prediction = AVG_RATING + dev_u + dev_i
		
			baselines.write(str(prediction) + "\n")
			error.append(abs(prediction - r) * abs(prediction - r))
		
			user += 1
			if (user % 100000 == 0):
				print(str(user) + " data points, RMSE: " + str(math.sqrt(np.mean(error))) + "")
			
	rmse = math.sqrt(np.mean(error))
	print("RMSE: " + str(rmse))

	baselines.close()
	
	print("Finished generating baseline predictions.")

generate_baselines()
