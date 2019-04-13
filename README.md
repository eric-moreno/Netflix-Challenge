# chardsdisciples
156b

bellkor_baseline.py
  - computes predictions of unknown movie ratings using the bellkor baseline formula: b = u + b_u + b_i
  - outputs predictions to 2d np-matrix "predictions.npy" (does not include training data values)
  

avg_ratings.py
  - computes average ratings and average deviations for all movies
  - stored in 2d np-matrix "avg_dev.npy"
      -- note : first row is [overall_average_movie_rating (u), 0.0]
      -- note : other rows are [average rating of movie, average deviation of movie (b_i)]
          -- to access movie i info, access avg_dev[i+1]
  - helper to bellkor_baseline.py

std_dev_predicty.py
  - method: prediction = (avg_movie_rating) + (user_std_dev)
  - generates a new text file "predictions.txt" with the unknown movie values of the training set predicted.
*** TO DO : Since I currently do not have access to the actual full matrix data, I used a dummy data set .txt to program the script, called "full_matrix.txt" nor did I have the actual movie averages (though these will be easy to compute) so I used a dummy .txt set called "averages.txt". Both of these will need to be modified to run for the actual data (see top of file for filenames). ***

full_array_train.npy (too large to store on github see initialization.py to run) 
  - full matrix of users (rows), movies (cols), and ratings (elements)
      -- integer : rating
      -- 0 : no data or validation data

Inside initialization.ipynb, the "movies" and "users" arrays are simply arrays for each movie's/user's corresponding scores.  

averages_method.ipynb
  - used for computing full matrix of users (rows), movies (cols), and ratings (elements)
      -- integer : rating
      -- 0 : no data or validation data
  
svd_scratch
  - SVD implemented from scratch, no external package used.
