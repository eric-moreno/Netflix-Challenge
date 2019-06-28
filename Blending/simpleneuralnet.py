# Neural network blending of baselines

import numpy as np
import pandas as pd
import math
import random

N_USERS = 480189

print ("Loading base data... \n")

base = pd.read_table('base.dta', delim_whitespace=True,header=None)
base = np.array(base)
	
b0 = pd.read_table('Baselines/baselines0.dta', delim_whitespace=True,header=None)
b0 = np.array(b0)
	
print("Finished loading baselines_0...\n")

b1 = pd.read_table('Baselines/baselines1.dta', delim_whitespace=True,header=None)
b1 = np.array(b1)

print("Finished loading baselines_1...\n")

b2 = pd.read_table('Baselines/baselines2.dta', delim_whitespace=True,header=None)
b2 = np.array(b2)

print("Finished loading baselines_2...\n")

b3 = pd.read_table('Baselines/baselines3.dta', delim_whitespace=True,header=None)
b3 = np.array(b3)

print("Finished loading baselines_3...\n")


print("Loading training examples...\n")

# input dataset
X = []
Y = []

'''for i in base:
	# user, movie, time, rating
	u, m, t, r =  i[0], i[1], i[2], i[3] 
		
	if (u < 17771): 
		X.append([b0[u], b1[u], b2[u], b3[u]])
		Y.append(r)'''

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = []
Y = []
# input dataset
u = base[0][0]
X.append([b0[u], b1[u], b2[u], b3[u]])

u = base[1][0]
X.append([b0[u], b1[u], b2[u], b3[u]])

u = base[2][0]
X.append([b0[u], b1[u], b2[u], b3[u]])

u = base[3][0]
X.append([b0[u], b1[u], b2[u], b3[u]])


X = np.array(np.squeeze(X))
print(X)
y = np.array([[base[0][3], base[1][3], base[2][3], base[3][3]]]).T

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((4,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

#pred = np.dot(l2)
print ("Ratings:\n" + str(y) + "\n")

print (str(l1) + "\n\n");
print (str(l2) + "\n");

