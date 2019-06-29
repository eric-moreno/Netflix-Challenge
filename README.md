Classwork for CS156b - Learning Systems (Caltech)
====================================================================

Introduction
===================================================================
Code used to get 7.7223% "above water" (above the original netflix classifier) with a quiz RMSE of 0.87793 contributing to an aggregate class performance of 8.9863% above water

Code is grouped in the different architectures used (SVD, KNN, Baselines, etc.). Outputs from these different architectures were then blended in various ways (NNs, regressions, averages) using the Blending folder. 

Descriptions (mostly used for us)
====================================================================

bellkor_baseline.py
  - reads training data from "base.npy" (see initialize_base.py)
  - computes predictions of unknown movie ratings using the bellkor baseline formula: b = u + b_u + b_i
  - outputs predictions to 2d np-matrix "predictions.npy" (does not include training data values)
      - format: [user_index, movie_index, predicted_rating]

initialize_base.py
  - read in training data into an np array "base.npy"
      - all data in "all.dta" that is indexed 1 in "all.idx"
  - helper to bellkor_baseline.py

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
