import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
import time
NUM_USERS = 458293
NUM_MOVIES = 17770
AVERAGE = 3.608608901826221

@njit
def predict(predict_on, ratings, similarity, k, predictions, user_dev, movie_avg):
    for row in range(len(predict_on)):
        user = predict_on[row][0] - 1
        movie = predict_on[row][1] - 1
        
        # Check if rating already exists
        if ratings[movie][user] != 0:
            predictions[row] = ratings[movie][user]
            
        # Otherwise, weighted averaged of K neighbors
        else:
            # We're not going to count the similarity with itself (1)
            similarity[movie][movie] = -2
            movie_row = similarity[movie]
            
            # Sorting trick to get indices of top k similar movies (in reverse order)
            indices = movie_row.argsort()[-k:]
            
            # New rating: add to baseline
            # sum of similarity * (rating - movie_avg) for all k items
            # divided by sum of similarities
            numer = 0
            denom = 0
            for i in range(1, len(indices)+1):
                # Go in order
                neigh = indices[-i]
                
                # Didn't rate it, we're out of useful neighbors
                if ratings[neigh][user] == 0:
                    break
                
                s_ij = similarity[movie][neigh]
                
                # Similarity * (rating - baseline)
                numer += s_ij * (ratings[neigh][user] - movie_avg[neigh])
                denom += s_ij
                
            # No similar movies...
            if denom == 0:
                pass
            else:
                predictions[row] -= 0.5 * numer/denom
    return predictions

@njit
def initial_matrix(train, A):
    print("Filling initial ratings")
    for i in range(len(train)):
        user_id = train[i][0]
        movie_id = train[i][1]
        rating = train[i][3]
        A[movie_id-1][user_id-1] = rating    
    return A

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
    
    # Congregate user, movie ratings
    A = np.zeros((NUM_MOVIES, NUM_USERS))
    initial_ratings = initial_matrix(train, A)
    
    print("loading similarity")
    #with h5py.File('similarity.h5', 'r') as hf:
        #sim_matrix = hf['similarities'][:]    
    with h5py.File('correlation.h5', 'r') as hf:
        sim_matrix = hf['correlation'][:]       
    
    print("loading devs")
    user_dev = np.loadtxt("user_dev.dta")
    movie_dev = np.loadtxt("movie_dev.dta")
    movie_avg = np.loadtxt("movie_avg.dta")
    
    print("making qual predictions")
    print("start time:", time.strftime("%X"))
    k = 100
    initial_predict = np.array(pd.read_table('qual_base_adjust.dta', delim_whitespace=True, header=None))
    qual_predict = predict(qual, initial_ratings, sim_matrix, k, initial_predict, user_dev, movie_avg)
    
    print("end time:", time.strftime("%X"))
    print("saving qual")
    np.savetxt("qual_predictions18.dta", qual_predict)
    
    print("making probe predictions")
    print("start time:", time.strftime("%X"))
    initial_predict = np.array(pd.read_table('probe_base_adjust.dta', delim_whitespace=True, header=None))
    probe_predict = predict(probe, initial_ratings, sim_matrix, k, initial_predict, user_dev, movie_avg)  

    print("end time:", time.strftime("%X"))
    print("saving probe")
    np.savetxt("probe_predictions18.dta", probe_predict)

if __name__ == '__main__':
    main() 