import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,3,5,7,9,13,20,20,21,24,26],[5,7,11,14,15,17,18,19,21,22,26]])
x = x.T

x_meaned = x-np.mean(x,axis=0)

plt.figure
plt.scatter(x_meaned[:,0],x_meaned[:,-1])
plt.scatter(x[:,0],x[:,-1])
plt.show()

# Using NumPy
#covariance matrix
c= np.cov(x_meaned, rowvar=False)
eval, evac = np.linalg.eig(c)
sorted_index = np.argsort(eval)[::-1]
sorted_eval = eval[sorted_index]
sorted_evec = evac[:,sorted_index]
n=1
evec_subset = sorted_evec[:,0:n]
#print(evec_subset.real)
x_reduced = np.dot(evec_subset.transpose(),x_meaned.transpose()).transpose()
print(x_reduced.real)

# Using scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(x_meaned)
xr = pca.transform(x_meaned)
print(xr)