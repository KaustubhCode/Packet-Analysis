import numpy as np
import pandas as pd
import sys

file = sys.argv[1]
outlier_limit = int(sys.argv[2])

df = pd.read_csv(file, header=None).astype('float')
df = df[df.iloc[:,:] < outlier_limit]

np.savetxt(file,df.dropna().values,delimiter=",")
