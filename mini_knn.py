import numpy as np
import pandas as pd
from numba import njit
from numba import jit
NUM_USERS = 8
NUM_MOVIES = 5

@njit
def initial_matrix(train, A):
    print("Filling initial ratings")
    for i in range(len(train)):
        user_id = train[i][0]
        movie_id = train[i][1]
        #!!! Change rating index
        rating = train[i][2]
        A[movie_id-1][user_id-1] = rating    
    return A

def similarity_matrix(A, B):
    '''Fills movie-movie similarity matrix B given initial sparse rating matrix A'''
    print("Populating similarities")
    for x in range(NUM_MOVIES):
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
    # Congregate user, movie ratings
    print("starting!")
    initial_ratings = np.array([[5,5,5,5,4],
                                [0,2,0,1,0],
                                [0,0,0,0,4],
                                [0,5,0,3,0],
                                [0,0,3,0,2],
                                [5,0,0,1,0],
                                [3,0,0,0,0],
                                [3,0,0,4,0]])
    print(initial_ratings)
    # Calculate cosine similarity for every movie-movie pair
    B = np.zeros((NUM_MOVIES, NUM_MOVIES))
    pair_similarities = similarity_matrix(initial_ratings, B)
    
    print("saving similarities")
    print(pair_similarities)
    
    predict_on = np.array([[1,3],
                         [4,2],
                         [7,5],
                         [8,1],
                         [5,1]])
    
    initial_predict = np.zeros((5, 1))
    print("predicting")
    predictions = predict(predict_on, initial_ratings, pair_similarities, 6, initial_predict)
    print(predictions)

if __name__ == '__main__':
    main() 