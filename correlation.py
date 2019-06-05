import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import time
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770

@njit
def initial_matrix(train):
    print("Filling initial ratings")
    A = np.zeros((NUM_USERS, NUM_MOVIES))
    for i in range(len(train)):
        user_id = train[i][0]
        movie_id = train[i][1]
        rating = train[i][3]
        A[user_id-1][movie_id-1] = rating    
    return A

@njit
def get_correlation(train, movie_avg, ratings):
    correlation = np.zeros((NUM_MOVIES, NUM_MOVIES))
    for i in range(NUM_MOVIES):
        if i % 1000 == 0:
            print("iteration:", i)
        for j in range(NUM_MOVIES):
            
            numer = 0
            denom1 = 0
            denom2 = 0
            for user in range(NUM_USERS):
                if ratings[user][i] != 0 and ratings[user][j] != 0:
                    numer += (ratings[user][i] - movie_avg[i]) * (ratings[user][j] - movie_avg[j])
                    denom1 += (ratings[user][i] - movie_avg[i]) * (ratings[user][i] - movie_avg[i])
                    denom2 += (ratings[user][j] - movie_avg[j]) * (ratings[user][j] - movie_avg[j])
                    
            if denom1 == 0 or denom2 == 0:
                pass
            else:
                # Pearson correlation
                correlation[i][j] = numer / (np.sqrt(denom1) * np.sqrt(denom2))
    print(correlation[:5])
    return correlation

@njit
def new_correlation(train, movie_avg, ratings):
    # for each position, store tuple of (num, denom1, denom2)
    numer = np.zeros((NUM_MOVIES, NUM_MOVIES))
    denom1 = np.zeros((NUM_MOVIES, NUM_MOVIES))
    denom2 = np.zeros((NUM_MOVIES, NUM_MOVIES))
    correlation = np.zeros((NUM_MOVIES, NUM_MOVIES))
    
    # for each movie
    for i in range(NUM_MOVIES):
        if i % 1000 == 0:
            print("iteration: ", i)
            
        # get user indices for ith column, shape (num watched users,)
        user_lst = np.where(ratings[:,i] != 0)
        
        # for users who rated the movie
        for index in range(len(user_lst)):
            user = user_lst[0][index]
            
            # get movie the user has watched, shape (num watched movies, )
            movie_lst = np.where(ratings[user] != 0)
            
            for index in range(len(movie_lst)):
                j = movie_lst[0][index]
                numer[i][j] += (ratings[user][i] - movie_avg[i]) * (ratings[user][j] - movie_avg[j])
                denom1[i][j] += (ratings[user][i] - movie_avg[i]) * (ratings[user][i] - movie_avg[i])
                denom2[i][j] += (ratings[user][j] - movie_avg[j]) * (ratings[user][j] - movie_avg[j])   
    
    print("generating correlation matrix")
    for i in range(NUM_MOVIES):
        for j in range(NUM_MOVIES):
            if denom1[i][j] == 0 or denom2[i][j] == 0:
                pass
            else:
                correlation[i][j] = numer[i][j] / (np.sqrt(denom1[i][j]) * np.sqrt(denom2[i][j]))
    return correlation


    
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

    # Congregate user, movie ratings
    ratings = initial_matrix(train)
    
    # Get avgs
    print("loading movie avgs")
    movie_avg = np.loadtxt("movie_avg.dta")
    print(movie_avg[:5])

    # Calculate pearson similarity for every movie-movie pair
    print("getting pearson correlation")
    print("!!!!!!!!!!!!!!!!!START:", time.strftime("%X"))
    correlation = new_correlation(train, movie_avg, ratings)
    
    print("!!!!!!!!!!!!!!!!!!DONE:", time.strftime("%X"))
    print("saving similarities")
    print(correlation[:3])
    print(sum(correlation[0]))
    with h5py.File('correlation.h5', 'w') as hf:
        hf.create_dataset("correlation",  data=correlation)
    

if __name__ == '__main__':
    main() 