import numpy as np

print("***************loading qual**************")
qual = np.load('qual.npy')
print(qual[:3])

print("------------loading U------------")
U = np.load('U.npy')
print(U[:4])

print("-----------loading V------------")
V = np.load('V.npy')
print(V[:4])

print("WRITING!!!!!")
f= open("predictions.dta","w+")
for row in range(qual.shape[0]):
    u = qual[row][0]
    i = qual[row][1]
    prediction = np.dot(U[u-1], V[i-1]) 
    string = str('%.3f'%(prediction)) + '\n'
    f.write(string)
print("********************MADE IT TO THE END!")
f.close 
print("CLOSING****************")