import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770

@njit
def movie_deviation(average, data):
    # Array with a row for each movie. 
    # 1st column running rating total, 2nd column rating count 
    movies = np.zeros((NUM_MOVIES, 2))
    
    print("Tallying scores")
    for i in range(len(data)):
        movie_id = data[i][1]
        rating = data[i][3]
        movies[movie_id-1][0] += rating
        movies[movie_id-1][1] += 1
    
    print("Generating averages")
    for j in range(NUM_MOVIES):
        if movies[j][1] == 0:
            movies[j][0] = 0
        else:
            movies[j][0] /= movies[j][1]
    
    print("Slicing movie devs")
    movie_dev = movies[:,0] - average
    return movie_dev

@njit
def user_deviation(average, data):
    # Array with a row for each movie. 
    # 1st column running rating total, 2nd column rating count 
    users = np.zeros((NUM_USERS, 2))
    
    print("Tallying scores")
    for i in range(len(data)):
        user_id = data[i][0]
        rating = data[i][3]
        users[user_id-1][0] += rating
        users[user_id-1][1] += 1
    
    print("Generating averages")
    for j in range(NUM_USERS):
        if users[j][1] == 0:
            users[j][0] = 0
        else:
            users[j][0] /= users[j][1]
    
    print("Slicing user devs")
    user_dev = users[:,0] - average
    return user_dev

@njit
def average_rating(data):
    ratings = data[:,3]
    total = ratings.sum()
    average = total/len(data)
    return average

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
    
    print("getting average")
    average = average_rating(train)
    print("average is:", average)
    
    movie_dev = movie_deviation(average, train)
    print("saving movie devs")
    print(movie_dev[:5])
    np.savetxt("movie_dev.dta", movie_dev) 
    
    user_dev = user_deviation(average, train)
    print("saving user devs")
    print(user_dev[:5])
    np.savetxt("user_dev.dta", user_dev)   
    
    print("!!!!!!!!!!!!!!!!!testing components!!!!!!!!!!!!!!!!!!!")
    print("actual rating:", train[16][3])
    print("user id:", train[16][0])
    print("movie id:", train[16][1])

    print("user dev:", user_dev[train[16][0]-1])
    print("movie dev:", movie_dev[train[16][1]-1])
    print("average:", average)
    
    movie_avg = get_movie_avg(train)
    print("movie average:", movie_avg[train[16][1]-1])
    print("calculated rating:", average + user_dev[train[16][0]-1] + movie_dev[train[16][1]-1])


if __name__ == '__main__':
    main() 
