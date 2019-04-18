import numpy as np
import pandas as pd
import time
from sklearn.utils.extmath import randomized_svd

def generate_predictions(U, VT, qual):
    '''Given training output decomposition U and V, output predictions
       for qual submission set.'''
    
    f= open("predictions.dta","w+")
    for row in qual:
        i = row[0]
        j = row[1]
        prediction = np.matmul(U[i-1], VT[j-1]) 
        string = str('%.3f'%(prediction)) + '\n'
        f.write(string)
    f.close 

# Load dataset as matrix of tuples (userid, movieid, time, rating)
print('loading dataset')
dataset = pd.read_table('all.dta', delim_whitespace=True, header=None)
data = np.array(dataset)
print(data[:5]) 

# Load indices for splitting
print('loading indices')
idx_set = pd.read_table('all.idx', delim_whitespace=True, header=None)
idx = np.array(idx_set)    
print('Splitting sets')

# Training set
rows, cols = np.where(idx == 1)
train = data[rows]

# qual set to predict on
rows, cols = np.where(idx == 5)
qual = data[rows]

# Train! With k = 64 latent factors for now
k = 64
start = time.time()
U, Sigma, VT = randomized_svd(train, k, n_iter=4)
end = time.time()
print("sklearn: %2.2f" % (end - start))

# Write predictions to a file
generate_predictions(U, VT, qual)

