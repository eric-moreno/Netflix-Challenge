# Blend baseline predictions

import numpy as np
import pandas as pd
import math

b0 = pd.read_table('Baselines/baselines0.dta', delim_whitespace=True,header=None)
b0 = np.array(b0)
	
print("Finished loading baselines_0...\n")

b1 = pd.read_table('Baselines/baselines1.dta', delim_whitespace=True,header=None)
b1 = np.array(b1)

print("Finished loading baselines_1...\n")

b2 = pd.read_table('Baselines/baselines2.dta', delim_whitespace=True,header=None)
b2 = np.array(b2)

print("Finished loading baselines_2...\n")

b3 = pd.read_table('Baselines/baselines3.dta', delim_whitespace=True,header=None)
b3 = np.array(b3)

print("Finished loading baselines_3...\n")

file = open("averaged2.dta", "w")
i = 0
for row in b0:
	file.write(str(np.mean([b0[i], b1[i], b2[i], b3[i]])) + "\n")
	
	if (i % 1000000 == 0) :
		print("Done with " + str(i) + " users. \n")
	
	i += 1;
	
file.close()

print("Finished averaging baselines!\n")
