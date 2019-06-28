import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
LEN_PROBE = 1374739
probe = "mu_probe.csv"
qual = "mu_qual.csv"
NN = True
linreg = False

import os
try:
    import setGPU
except:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Models

# Add filename of probe predictions to list below
probe_models = ["Time_SVD_probe_preds_200_factors.txt", "SVD++_probe_preds_100_factors.txt", "RBM_probe.txt", "PMF_probe.txt", "richard_blend1probe.dta", "richard_blend2probe.dta", "richard_blend3probe.dta", "richard_blend4probe.dta", "richard_blend5probe.dta", "richard_blend6probe.dta", "richard_blend9probe.dta", "richard_blend11probe.dta", 'richard_singlesol1probe.dta', 'richard_singlesol2probe.dta', 'neighborhood_chard_probe1.dta']
#probe_models = ["richard_blend1probe.dta", "richard_blend2probe.dta", "richard_blend3probe.dta", "richard_blend4probe.dta", "richard_blend5probe.dta", "richard_blend6probe.dta", "richard_blend9probe.dta", "richard_blend11probe.dta", 'richard_singlesol1probe.dta', 'richard_singlesol2probe.dta', 'neighborhood_chard_probe1.dta']
#probe_models = ["Time_SVD_probe_preds_200_factors.txt", "SVD++_probe_preds_100_factors.txt", "RBM_probe.txt", "PMF_probe.txt", "richard_blend9probe.dta", "richard_blend11probe.dta",'neighborhood_chard_probe1.dta', 'POLY2probe.dta', 'probe_predictions_laura.dta', 'LIN1probe.dta', 'timeSVD4probe.dta']
qual_models = ["Time_SVD_preds_100_factors_probetrained.txt", "SVD++_preds_100_factors_probetrained.txt", "RBM_qual.txt", "PMF_qual.txt", "richard_blend1.dta", "richard_blend2.dta", "richard_blend3.dta", "richard_blend4.dta", "richard_blend5.dta", "richard_blend6.dta", "richard_blend9.dta", "richard_blend11.dta", "richard_singlesol1.dta",  "richard_singlesol2.dta", 'neighborhood_chard_1.dta']
#qual_models = ["Time_SVD_preds_100_factors_probetrained.txt","SVD++_preds_100_factors_probetrained.txt", "RBM_qual.txt", "PMF_qual.txt" , "richard_blend9.dta", "richard_blend11.dta", 'neighborhood_chard_1.dta', 'POLY2.dta', 'qual_predictions_laura.dta', 'LIN1.dta', 'timeSVD4.dta']
#qual_models = ["richard_blend1.dta", "richard_blend2.dta", "richard_blend3.dta", "richard_blend4.dta", "richard_blend5.dta", "richard_blend6.dta", "richard_blend9.dta", "richard_blend11.dta", "richard_singlesol1.dta",  "richard_singlesol2.dta", 'neighborhood_chard_1.dta']

if len(probe_models) != len(qual_models):
    raise ValueError("Must have qual and probe predictions for all models")

# Loop over the models to read in the training data
X = []
for model in probe_models:
    data = [line for line in open(model)]
    if len(data) < LEN_PROBE:
        raise ValueError("Length of input should be equal to length of probe dataset")
    X.append(data)
X = np.array(X).T
print(X)


y = []
for line in open(probe): 
    if len(line) < 4: 
        continue
    else: 
        y.append(int(line.split(",")[3]))
y = np.array(y)
print(len(y))
# Read in probe data (Training data)
#y = np.array([int(line.split(",")[3]) for line in open(probe)])

# Read in model qual data
test = []
for model in qual_models:
    data = [line for line in open(model)]
    test.append(data)
test = np.array(test).T

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
if NN:
    # Model
    model = Sequential()
    model.add(Dense(25, input_dim=len(probe_models)))
    model.add(Dense(15))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss=root_mean_squared_error, metrics=["accuracy"])
    model.summary()
    
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    mcp_save = ModelCheckpoint('best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(X, y, epochs=30, batch_size=128, validation_split = 0.1, callbacks = [mcp_save])

    # Predict on qual set
    print("Writing predictions")
    preds = model.predict(test)
    preds_probe = model.predict(X)

    model.load_weights('best.hdf5')
    preds_best = model.predict(test)
    preds_best_probe = model.predict(X)
    for i in range(len(preds_best)):
      if preds_best[i] < 1:
          preds_best[i] = 1
      elif preds_best[i] > 5:
          preds_best[i] = 5
    file = open("fullblend2.txt", "w")
    file.writelines(["%s\n" % item[0] for item in preds_best])
    
    for i in range(len(preds_best_probe)):
      if preds_best_probe[i] < 1:
          preds_best_probe[i] = 1
      elif preds_best_probe[i] > 5:
          preds_best_probe[i] = 5
    file = open("full2probe.txt", "w")
    file.writelines(["%s\n" % item[0] for item in preds_best_probe])

    
elif linreg: 
    print('Running Linear Regression') 
    from sklearn.linear_model import LinearRegression
    
    for i in range(len(X)): 
        X[i] = int(X[i][:-2])
    reg = reg = LinearRegression().fit(X, y)
    print(X[:10]) 
    print(y[:10])
    preds = []
    for i in test:
        print(i)
        print(reg.predict([i]))
        preds.append(reg.predict([i]))
else: 
    print('Running Ridge Regression') 
    from sklearn.linear_model import Ridge
    
    for i in range(len(X)): 
        X[i] = float(X[i][:-2])
    
    '''
    for i in range(len(X)): 
        arr = []
        for j in range(len(X[i])): 
            print(X[i][j][:-2])
            if len(X[i][j][:-2]) == 1: 
                arr.append(3)
                print('missing')
            else: 
                arr.append(float(X[i][j][:-2]))
        X[i] = arr   
        
    '''
    print(X)
    reg = reg = Ridge().fit(X, y)
    print(X[:10]) 
    print(y[:10])
    preds = []
    for i in test:
        print(i)
        print(reg.predict([i]))
        preds.append(reg.predict([i]))
   
# Cap the ratings
for i in range(len(preds)):
  if preds[i] < 1:
      preds[i] = 1
  elif preds[i] > 5:
      preds[i] = 5
file = open("fullblend3.txt", "w")
file.writelines(["%s\n" % item[0] for item in preds])
