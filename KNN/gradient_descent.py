import numpy as np
import pandas as pd
from numba import njit
from numba import jit
NUM_USERS = 458293
NUM_MOVIES = 17770

@njit
def SGD(data, n_factors, alpha, reg, n_epochs):
    '''Learn the vectors p_u and q_i with SGD.
       data is a dataset containing all ratings + some useful info (e.g. number
       of items/users).
       n_factors - number of latent factors
       alpha - learning rate
       reg - regularization factor
       n_epochs - number of iterations
    '''
    
    # Randomly initialize the user and item factors.
    U = np.random.normal(0, .1, (NUM_USERS, n_factors))
    V = np.random.normal(0, .1, (NUM_MOVIES, n_factors))

    # Optimization procedure
    for epoch in range(n_epochs):
        print("epoch: ", epoch)
        for row in range(data.shape[0]):
            u = data[row][0] - 1
            i = data[row][1] - 1
            r_ui = data[row][3]
            err = r_ui - np.dot(U[u], V[i])
            
            # Update vectors U_u and V_i
            U[u] += alpha * (err * V[i] - reg * U[u])
            V[i] += alpha * (err * U[u] - reg * V[i])
            
    return (U, V)
            
def generate_predictions(U, V, qual):
    '''Given training output decomposition U and V, output predictions
       for qual submission set.'''
    print("!!!!!!!WRITING PREDICTIONS!!!!!!!!!!!")
    f= open("predictions.dta","w+")
    for row in range(qual.shape[0]):
        u = qual[row][0] - 1
        i = qual[row][1] - 1
        prediction = np.dot(U[u], V[i]) 
        string = str('%.3f'%(prediction)) + '\n'
        f.write(string)
    f.close 


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
    print("SAVING QUAL")
    np.save("qual", qual)

    # Train! With K = 64 latent factors for now
    n_factors = 100  # number of factors
    alpha = .002  # learning rate
    n_epochs = 25  # number of iteration of the SGD procedure 
    reg = 0.1 # regularization factor
    print('starting to train')
    U, V = SGD(train, n_factors, alpha, reg, n_epochs)
    print(U[0])
    print(V[0])
    
    # Write predictions to a file
    #generate_predictions(U, V, qual)
    print("FIRST SAVE!!!!!!!!!!!!!!!!!!!!!")
    np.save("U", U)
    print("SECOND SAVE!!!!!!!!!!!!!!!!!!!!!!")
    np.save("V", V)
    
    generate_predictions(U, V, qual)
    print("MADE IT TO THE END???")
    
if __name__ == '__main__':
    main()

