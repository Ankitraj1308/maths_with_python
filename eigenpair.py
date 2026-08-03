import numpy as np

A = np.array([[2,-2,3],[1,1,1],[1,3,-1]])
print(A)
values , vectors = np.linalg.eig(A)
print(values.real)
print(vectors.real) 

B = np.array([[2,0,0],[0,1,0],[0,0,-1]])
values2 , vectors2 = np.linalg.eig(B)
print(values2.real)
print(vectors2.real)