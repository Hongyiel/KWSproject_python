import numpy as np
import pandas as pd


data = pd.read_csv("data_set.csv",header=None)

print(np.shape(data))
print(data[0][0])

