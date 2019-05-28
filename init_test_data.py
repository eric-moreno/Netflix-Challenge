# Create test data for Bellkor Baseline predictor
import numpy as np

movie_matrix = [[3, 0.0, 0.0]]	
movie_matrix.append([1, 1.0, 450])
movie_matrix.append([5, .3, 12])
movie_matrix.append([2.5, 0.5, 634])
movie_matrix.append([4, 0.5, 54])

movie_matrix = np.array(movie_matrix)
np.save('movie_test.npy', movie_matrix) 

b_u = [-0.5, 0.5, 0.0, -0.5, 1.5, -0.3, 0.3, 0.8, 0.8]

user_matrix = np.array(b_u)
np.save('user_test.npy', user_matrix) 