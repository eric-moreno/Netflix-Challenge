import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

folder = 'E:\\Documents\\Caltech\\CS156\\BLEND\\'
bfolder = 'E:\\Documents\\Caltech\\CS156\\'


X = []
qmean = 3.674

#baseSol = pd.read_csv(folder + 'LIN1.dta', delim_whitespace=True, header=None)
#useSols = ['blended_chard.txt', 'blended_chard_best.txt', 'blended_chard_chard_best.txt', 'timeSVD.dta', 'POLY.dta', 'timeSVD2.dta', 'POLYrbm.dta']
#errors = [0.89051, 0.8815, 0.88069, 0.88075, 0.89968, 0.88922, 0.89985, 0.88052, 0.89484]
#useSols = ['blended_chard.txt', 'blended_chard_best.txt', 'blended_chard_chard_best.txt', 'timeSVD.dta', 'POLY2.dta', 'timeSVD2.dta', 'POLYrbm.dta', 'timeSVD4.dta', 'new_averages.dta']
#errors = [0.89051, 0.8815, 0.88069, 0.88075, 0.89968, 0.8885, 0.89985, 0.88052, 0.89347, 0.87809]
baseSol = pd.read_csv(folder + 'blended_chard.txt', delim_whitespace=True, header=None)
useSols = ['blended_chard_best.txt', 'blended_chard_chard_best.txt', 'new_averages.dta', 'POLY2.dta', 'QBlended.dta']
errors = [0.8815, 0.88069, 0.88075,  0.87809, 0.8885, 0.87795]

for i in useSols:
    filename = folder + i
    sol = pd.read_csv(filename, delim_whitespace=True, header=None)
    baseSol = pd.concat([baseSol, sol], axis=1)

Xa = baseSol.values

X = np.asarray(Xa)
X = X - qmean

print(X)
N = len(X)
p = len(X[0])

Y = []

for i in range(p):
    sum = 0
    for j in range(N):
        sum += X[j][i] ** 2
    curr = 0.5 * (N * 1.274 + sum - N * errors[i] * errors[i])
    Y.append([curr])

S1 = (np.matmul(np.transpose(X), X))
RR = 0.0014 * N * np.identity(p)
S1 += RR
S1 = np.linalg.inv(S1)
B = np.matmul(S1, Y)
print(B)
#X = X + qmean

Q = np.matmul(X, B)
Q += qmean
print(Q)
print(1.274 - np.var(Q))
print("DONE")
np.savetxt(folder + 'Qblended.dta', Q, fmt='%1.5f')
