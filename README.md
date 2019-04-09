# chardsdisciples
156b

std_dev_predicty.py
  - method: prediction = (avg_movie_rating) + (user_std_dev)
  - generates a new text file "predictions.txt" with the unknown movie values of the training set predicted.
*** TO DO : Since I currently do not have access to the actual full matrix data, I used a dummy data set .txt to program the script, called "full_matrix.txt" nor did I have the actual movie averages (though these will be easy to compute) so I used a dummy .txt set called "averages.txt". Both of these will need to be modified to run for the actual data (see top of file for filenames). ***

full_array_train.npy
  - full matrix of users (rows), movies (cols), and ratings (elements)
      -- integer : rating
      -- 0 : no data or validation data
      
averages_method.ipynb
  - used for computing full matrix of users (rows), movies (cols), and ratings (elements)
      -- integer : rating
      -- 0 : no data or validation data
  
svd_scratch
  - off-shelf SVD model
