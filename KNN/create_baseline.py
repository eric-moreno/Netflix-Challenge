import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770
AVERAGE = 3.608608901826221

@njit
def generate_baseline(data, user_dev, movie_dev):
    result = np.zeros((len(data), 1))
    for i in range(len(data)):
        user = data[i][0]
        movie = data[i][1]
        result[i] = AVERAGE + user_dev[user] + movie_dev[movie]
    return result

def main():
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
    
    # probe set to also predict on
    rows, cols = np.where(idx == 4)
    probe = data[rows]
        
    print("loading devs")
    user_dev = np.loadtxt("user_dev.dta")
    movie_dev = np.loadtxt("movie_dev.dta")
    
    print("saving qual_base")
    qual_base = generate_baseline(qual, user_dev, movie_dev)
    np.savetxt("qual_base.dta", qual_base)
    
    print("saving probe_base")
    probe_base = generate_baseline(probe, user_dev, movie_dev)
    np.savetxt("probe_base.dta", probe_base)

if __name__ == '__main__':
    main() 
