import numpy as np
np. set_printoptions(precision=4,suppress=True)

a = np.array([[1,2,3,4],[1,1,2,3],[0,1,1,0]])
u, s, vh = np.linalg.svd(a,full_matrices=True)
sd = np.diag(s)
b = np.zeros((3,4))
b[:,:-1]=sd
sigma = b
svd = np.dot(np.dot(u,sigma),vh)
print(u)
print(sigma)
print(vh)
print(svd)