import numpy as np
import os.path
import sys
import csv

# A = [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
# B = [[[2,3,3],[1,2,3]],[[2,3,3],[1,2,3]]]


np.random.seed(42)
  
A = np.random.randint(0, 10, size=(3, 3, 2))
B = np.random.randint(0, 10, size=(3, 2, 4))

print("A:\n{}, shape={}\nB:\n{}, shape={}".format(
  A, A.shape, B, B.shape))


print(np.shape(A))
print(np.shape(B))

C = np.matmul(A, B)

print("Product C:\n{}, shape={}".format(C, C.shape))
