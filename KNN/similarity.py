import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770

@njit
def initial_matrix(train, A):
    print("Filling initial ratings")
    for i in range(len(train)):
        user_id = train[i][0]
        movie_id = train[i][1]
        rating = train[i][3]
        A[movie_id-1][user_id-1] = rating    
    return A

@njit
def similarity_matrix(A, B):
    '''Fills movie-movie similarity matrix B given initial sparse rating matrix A'''
    print("Populating similarities")
    for x in range(NUM_MOVIES):
        if x % 100 == 0:
            print(x)
        for y in range(NUM_MOVIES):
            # Matrix is symmetrical, so copy transpose value if it exists
            if (B[y][x] != 0):
                B[x][y] = B[y][x]
            else:
                numer = np.vdot(A[x], A[y])
                denom = np.linalg.norm(A[x]) * np.linalg.norm(A[y])
                B[x][y] =  numer/denom
    return B

@njit
def predict(predict_on, ratings, similarity, k, predictions):
    '''Make predictions on input given rating matrix A, similarity B, k neighbors, and
       empty initial predictions'''
    
    for row in range(len(predict_on)):
        user = predict_on[row][0] - 1
        movie = predict_on[row][1] - 1
        
        # Check if rating already exists
        if ratings[user][movie] != 0:
            predictions[row] = ratings[user][movie]
            
        # Otherwise, weighted averaged of K neighbors
        else:
            movie_row = similarity[movie]
            
            # Ignore neighbors whom I haven't rated
            for j in range(len(movie_row)):
                if ratings[user][j] == 0:
                    movie_row[j] = 0
            
            # Sorting trick to get indices of top k similar movies (in reverse order)
            indices = movie_row.argsort()[-k:]
            
            # Take weighted average
            numer = 0
            denom = 0
            for i in range(len(indices)):
                numer += ratings[user][indices[i]] * similarity[movie][indices[i]]
                denom += similarity[movie][indices[i]]
                
            # No similar movies...
            if denom == 0:
                print("no similar")
                predictions[row] = 3
            else:
                predictions[row] = numer/denom
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
    train = data[rows]
    '''
    # qual set to predict on
    rows, cols = np.where(idx == 5)
    qual = data[rows]
    
    # probe set to also predict on
    rows, cols = np.where(idx == 4)
    probe = data[rows]
    '''
    # Congregate user, movie ratings
    A = np.zeros((NUM_MOVIES, NUM_USERS))
    initial_ratings = initial_matrix(train, A)
    
    print("saving initial")
    with h5py.File('initial.h5', 'w') as hf:
        hf.create_dataset("initial",  data=initial_ratings)
    '''    
    # Calculate cosine similarity for every movie-movie pair
    B = np.zeros((NUM_MOVIES, NUM_MOVIES))
    pair_similarities = similarity_matrix(initial_ratings, B)
    
    print("saving similarities")
    print(pair_similarities[:3])
    with h5py.File('similarity.h5', 'w') as hf:
        hf.create_dataset("similarities",  data=p)
    
    print("making qual predictions")
    k = 50
    initial_predict = np.zeros((len(qual), 1))
    qual_predict = predict(qual, initial_ratings, pair_similarities, k, initial_predict)
    
    print("saving qual")
    np.savetxt("qual_predictions.dta", qual_predict)
    
    print("making probe predictions")
    initial_predict = np.zeros((len(probe), 1))
    probe_predict = predict(probe, initial_ratings, pair_similarities, k, initial_predict)  

    print("saving probe")
    np.savetxt("probe_predictions.dta", probe_predict)
    '''
    

if __name__ == '__main__':
    main() 
