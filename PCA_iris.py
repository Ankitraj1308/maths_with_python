from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()

pca = PCA(n_components=2)
pca.fit(iris.data)

x = pca.transform(iris.data)
plt.scatter(x[:, 0], x[:, 1], c=iris.target)
plt.scatter([x[10][0]], [x[10][1]], s=200, c="red", alpha=0.5)

plt.show()