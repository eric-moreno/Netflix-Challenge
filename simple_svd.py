import numpy as np
import pandas as pd

NUM_USERS = 458293;
NUM_MOVIES = 17770;

N_EPOCHS = 1000;		# epochs
ALPHA = 0.01; 			# learning rate
K = 64; 				# latent factors

# Randomly initialize the user and item factors.
p = np.random.normal(0, .1, (NUM_USERS, K))
q = np.random.normal(0, .1, (NUM_MOVIES, K))

def train():
	'''Learn the vectors p_u and q_i with SGD.
	data is a dataset containing all ratings + some useful info (e.g. number
	of items/users).
	'''
	base = open("base.txt", "r")
	
	# Optimization procedure
	for n in range(N_EPOCHS):
		for line in base:
			u, i, t, r = [int(x) for x in line.split()]
			err = r - np.dot(p[u], q[i])
			p[u] += ALPHA * err * q[i]
			q[i] += ALPHA * err * p[u]

def predict():
	'''Estimate rating of user u for item i.'''
	qual = open("qual.txt", "r")
	pred = open("predictions.txt", "w")
	
	for line in qual:
		u, i = [int(x) for x in line.split()]
		prediction = np.dot(p[u], q[i])
		pred.write(str(prediction)+ '\n')

def main():
	print('Begin training...')
	train()
	print('Finished training...')
	print('Begin predicting...')
	predict()
	print('Finished predicting...\n')
	
	print('All done!')
    
if __name__ == '__main__':
	main()
