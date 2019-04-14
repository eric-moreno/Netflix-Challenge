import numpy as np
from numba import jit
NUM_USERS = 458293
NUM_MOVIES = 17770

@jit
def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    grad = reg * Ui - Vj * (Yij - np.inner(Ui, Vj))  
    return eta * grad

@jit
def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    grad = reg * Vj - Ui * (Yij - np.inner(Ui, Vj))
    return eta * grad

@jit
def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, time Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    regularized = reg / 2 * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)

    # Summation of square error terms
    error = 0
    for i, j, time, y_ij in Y:
        error += (y_ij - np.matmul(U[i-1].T, V[j-1]))**2
    return regularized + 0.5 * error

@jit
def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=3000):
    """
    Given a training data matrix Y containing rows (i, j, time, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    print('training!')
    # Initialize U and V
    U = np.random.uniform(-0.5, 0.5, (M, K))
    V = np.random.uniform(-0.5, 0.5, (N, K))
    indices = list(range(len(Y))) 

    # Keep track of epochs
    epochs = 0

    # Keep track of errors for first epoch and last epoch
    first_improv = get_err(U, V, Y)
    last_error = first_improv

    while epochs < max_epochs:
        if epochs % 100 == 0:
            print("epoch:", epochs)
            
        np.random.shuffle(indices)
        epochs += 1

        # Update U: i is located at Y[n][0] and the j is located at Y[n][1]
        for n in indices:
            i = Y[n][0]
            j = Y[n][1]
            y_ij = Y[n][3] # time is stored at Y[n][2], which we ignore for now
            U[i-1] = U[i-1] - grad_U(U[i-1], y_ij, V[j-1].T, reg, eta)
            V[j-1] = V[j-1] - grad_V(V[j-1].T, y_ij, U[i-1], reg, eta)

        curr_error = get_err(U, V, Y)        
        if epochs == 1:
            first_improv -= curr_error

        # Check improvement condition
        elif (last_error - curr_error)/first_improv < eps:
            break
        last_error = curr_error 

    # Calculate error
    error = get_err(U, V, Y)
    return (U, V, error)

@jit
def generate_predictions(U, V, qual):
    '''Given training output decomposition U and V, output predictions
       for qual submission set.'''
    
    f= open("averages.dta","w+")
    for i, j, time, y_ij in qual:
        prediction = np.matmul(U[i-1].T, V[j-1]) 
        string = str('%.3f'%(prediction)) + '\n'
        f.write(string)
    f.close 


@jit
def main():
    # Load dataset as matrix of tuples (userid, movieid, time, rating)
    dataset = np.loadtxt("all.dta").astype(int) 
    
    # Load indices for splitting
    idx_set = pd.read_table('all.idx', header=None)
    idx = np.array(idx_set)    

    # Training set
    rows, cols = np.where(idx == 1)
    train = data[rows]
    
    # qual set to predict on
    rows, cols = np.where(idx == 2)
    qual = data[rows]

    # Train! With K = 64 latent factors for now
    K = 64
    reg = 0.0
    eta = 0.03    
    U, V, err = train_model(NUM_USERS, NUM_MOVIES, K, eta, reg, training)
    
    # Write predictions to a file
    generate_predictions(U, V, qual)
    
if __name__ == '__main__':
    main()