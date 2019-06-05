import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770

@njit
def get_movie_avg(train):
    movie_avg = np.zeros((NUM_MOVIES,2))
    
    for i in range(len(train)):
        movie = train[i][1] - 1
        rating = train[i][3]
        movie_avg[movie][0] += rating
        movie_avg[movie][1] += 1
    
    for j in range(NUM_MOVIES):
        movie_avg[j][0] /= movie_avg[j][1]
    
    return movie_avg[:,0]

@njit        
def generate_baseline(data, movie_avg, user_dev):
    result = np.zeros((len(data),))
    for i in range(len(data)):
        movie = data[i][1] - 1
        user = data[i][0] - 1
        result[i] = movie_avg[movie] + user_dev[user]
    return result


def main():
    # Load dataset as matrix of tuples (userid, movieid, time, rating)
    print('loading dataset')
    dataset = pd.read_table('all_mu.dta', delim_whitespace=True, header=None)
    data = np.array(dataset)
    print(data[:5]) 
    
    # Load indices for splitting
    print('loading indices')
    idx_set = pd.read_table('all_mu.idx', delim_whitespace=True, header=None)
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
    
    print("getting movie avgs")
    movie_avg = get_movie_avg(train)
    np.savetxt("movie_avg.dta", movie_avg)
    print(movie_avg[:5])
    
    print("loading devs")
    user_dev = np.loadtxt("user_dev.dta")
    
    print("saving qual_base")
    qual_base = generate_baseline(qual, movie_avg, user_dev)
    print(qual_base[:5])
    print("test point:", train[116])
    print("result:", movie_avg[train[116][1]-1] + user_dev[train[116][0]-1])
    np.savetxt("qual_base_adjust.dta", qual_base)
    
    print("saving probe_base")
    probe_base = generate_baseline(probe, movie_avg, user_dev)
    np.savetxt("probe_base_adjust.dta", probe_base)

if __name__ == '__main__':
    main() 