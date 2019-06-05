import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770
AVERAGE = 3.608608901826221

def initial_matrix(data, A):
    for i in range(len(data)):
        user_id = data[i][0]
        movie_id = data[i][1]
        rating = data[i][3]
        A[movie_id-1][user_id-1] = rating    
    return A


def train(data, ratings, similarity, weights, user_dev, movie_dev, k, lamb, gamma, num_iter):
    iters = 0
    
    while iters < num_iter:
        iters += 1
        print("iters:", iters)
        
        for i in range(len(data)):
            user = data[0] - 1
            movie = data[1] - 1
            rating = data[3]
            error = rating - ratings[movie][user]
            
            user_dev[user] += gamma * (error - lamb * user_dev[user])
            movie_dev[movie] += gamma * (error - lamb * movie_dev[movie])
            
            # Sorting trick to get indices of top k similar movies (in reverse order)
            movie_row = similarity[movie]
            indices = movie_row.argsort()[-k:] 
            
            for j in range(len(indices)):
                movie2 = indices[i]
                baseline = AVERAGE + user_dev[user] + movie_dev[movie2]
                weights[movie][movie2] += gamma * (error * (ratings[movie2][user] - baseline) - lamb * weights[movie][movie2])
    
    return user_dev, movie_dev, weights
            

def predict(predict_on, weights, ratings, similarity, k, user_dev, movie_dev):
    predictions = np.zeros((len(predict_on), 1))
    for i in range(len(predict_on)):
        user = predict_on[i][0] - 1
        movie = predict_on[i][1] - 1
        predictions[i] = AVERAGE + user_dev[user] + movie_dev[movie]
        
        # Check if rating already exists
        if ratings[movie][user] != 0:
            predictions[row] = ratings[movie][user]
            
        # Otherwise, add weighted averaged of K neighbors
        else:
            movie_row = similarity[movie]
            
            # Ignore neighbors whom I haven't rated
            for j in range(len(movie_row)):
                if ratings[user][j] == 0:
                    movie_row[j] = 0
            
            # Sorting trick to get indices of top k similar movies (in reverse order)
            indices = movie_row.argsort()[-k:]
            
            # Take weighted average
            total = 0
            for i in range(len(indices)):
                movie2 = indices[i]
                baseline = AVERAGE + user_dev[user] + movie_dev[movie2]
                total += weights[movie][movie2] * (ratings[user][movie2] - baseline)
            predictions[row] += total
                
    return predictions

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
    training = data[rows]
    
    # qual set to predict on
    rows, cols = np.where(idx == 5)
    qual = data[rows]
    
    # probe set to also predict on
    rows, cols = np.where(idx == 4)
    probe = data[rows]
    
    # Congregate user, movie ratings
    A = np.zeros((NUM_MOVIES, NUM_USERS))
    initial_ratings = initial_matrix(training, A)
    
    print("loading similarity")
    with h5py.File('similarity.h5', 'r') as hf:
        sim_matrix = hf['similarities'][:]    
    
    weights = np.copy(sim_matrix)
    
    print("loading devs")
    user_dev = np.loadtxt("user_dev.dta")
    movie_dev = np.loadtxt("movie_dev.dta")
    
    print("training")
    k = 300
    gamma = 0.005
    lamb = 0.002
    num_iter = 15
    
    bu, bi, wij = train(training, initial_ratings, sim_matrix, weights, user_dev, movie_dev, k, lamb, gamma, num_iter)
    
    print("saving weights")
    with h5py.File('weights.h5', 'w') as hf:
        hf.create_dataset("wij",  data=wij)    
    
    print("making qual predictions")
    qual_predict = predict(qual, wij, initial_ratings, sim_matrix, k, bu, bi)
    
    print("saving qual")
    np.savetxt("qual_predictions.dta", qual_predict)
    
    print("making probe predictions")
    probe_predict = predict(probe, wij, initial_ratings, sim_matrix, k, bu, bi)

    print("saving probe")
    np.savetxt("probe_predictions.dta", probe_predict)

if __name__ == '__main__':
    main() 