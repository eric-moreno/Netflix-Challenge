import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770

@njit
def get_A_bar(ratings):
    A_bar = np.zeros((NUM_MOVIES, NUM_MOVIES))
    for i in range(NUM_MOVIES):
        for j in range(NUM_MOVIES):
            numer = 0
            denom = 0
            for user in range(len(ratings)):
                if ratings[user][i] != 0 and ratings[user][j] != 0:
                    numer += ratings[user][i] * ratings[user][j]
                    denom += 1
            if denom == 0:
                pass
            else:
                A_bar[i][j] = numer/denom
    return A_bar

@njit
def get_b_bar(ratings):
    b_bar = np.zeros((NUM_MOVIES,))
    for i in range(NUM_MOVIES):
        numer = 0
        denom = 0
        for user in range(len(ratings)):
            if ratings[user][i] != 0:
                for j in range(NUM_MOVIES):
                    if ratings[user][j] != 0:
                        numer += ratings[user][i] * ratings[user][j]
                        denom += 1
        if denom == 0:
            pass
        else:
            b_bar[i] = numer/denom
    return b_bar

@njit
def get_avg(A_bar):
    diag_total = 0
    reg_total = 0
    total = 0
    for i in range(NUM_MOVIES):
        for j in range(NUM_MOVIES):
            total += A_bar[i][j]
            if i == j:
                diag_total += A_bar[i][j]
            else:
                reg_total += A_bar[i][j]
                
    diag_average = diag_total/NUM_MOVIES
    reg_average = reg_total/(NUM_MOVIES * NUM_MOVIES - NUM_MOVIES)
    average = total/(NUM_MOVIES * NUM_MOVIES)
    return diag_average, reg_average, average

@njit            
def get_A_hat(ratings, A_bar, diag_avg, reg_avg, beta):
    A_hat = np.zeros((NUM_MOVIES, NUM_MOVIES))
    for i in range(NUM_MOVIES):
        for j in range(NUM_MOVIES):
            count = 0
            for user in range(len(ratings)):
                if ratings[user][i] != 0 and ratings[user][j] != 0:
                    count += 1
                    
            denom = count + beta
            numer = count * A_bar[i][j]
            if i == j:
                numer += beta * diag_avg
            else:
                numer += beta * reg_avg
            A_hat[i][j] = numer/denom  
    return A_hat

@njit
def get_b_hat(ratings, b_bar, avg, beta):
    b_hat = np.zeros((NUM_MOVIES,))
    for i in range(NUM_MOVIES):
        count = 0
        for user in range(len(ratings)):
            if ratings[user][i] != 0:
                for j in range(NUM_MOVIES):
                    if ratings[user][j] != 0:
                        count += 1
                        
        numer = count * b_bar[i] + beta * avg
        denom = count + beta
        b_hat[i] = numer/denom
    return b_hat

@njit
def initial_matrix(data):
    ratings = np.zeros((NUM_USERS, NUM_MOVIES))
    for i in range(len(data)):
        user_id = data[i][0]
        movie_id = data[i][1]
        rating = data[i][3]
        ratings[user_id-1][movie_id-1] = rating    
    return ratings                  


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
    
    # Congregate user, movie ratings
    print("getting initial ratings")
    ratings = initial_matrix(train)
    
    # Compute b_bar for all movie pairs
    print("computing b_bar")
    b_bar = get_b_bar(ratings)
    
    print("saving b_bar")
    with h5py.File('b_bar.h5', 'w') as hf:
        hf.create_dataset("b_bar",  data=b_bar)    
    
    # Compute A_bar for all movie pairs
    print("computing A_bar")
    A_bar = get_A_bar(ratings)
    
    print("saving A_bar")
    with h5py.File('A_bar.h5', 'w') as hf:
        hf.create_dataset("A_bar",  data=A_bar)      
    
    # Compute averages
    print("computing averages")
    diag_avg, reg_avg, avg = get_avg(A_bar)
    
    # Compute b_hat
    beta = 500
    print("computing b_hat")
    b_hat = get_b_hat(ratings, b_bar, avg, beta)
    
    print("saving b_hat")
    with h5py.File('b_hat.h5', 'w') as hf:
        hf.create_dataset("b_hat",  data=b_hat)    
        
    # Compute  A_hat
    print("computing A_hat")
    A_hat = get_A_hat(ratings, A_bar, diag_avg, reg_avg, beta)
    
    print("saving A_hat")
    with h5py.File('A_hat.h5', 'w') as hf:
        hf.create_dataset("A_hat",  data=A_hat)    
        


if __name__ == '__main__':
    main() 
