import numpy as np
import pandas as pd
from numba import njit
from numba import jit
import h5py
NUM_USERS = 458293
NUM_MOVIES = 17770

def solve_quad(A, b, x):
    k = len(b)
    res = (A @ x) - b
    
    while res > 0.01:
        for i in range(k):
            if x[i] == 0 and res[i] < 0:
                res[i] = 0
                
        alpha = (res.T @ res) / (res.T @ A @ res)
        
        for i in range(k):
            if res[i] < 0:
                alpha = min(alpha, -x[i]/res[i])
        
        x += alpha * res
        res = (A @ x) - b
    return x

def get_A(k, similarities, A_hat):
    
