from sklearn.datasets import load_iris
from algs.ml_algorithms.mixgaussian import Gmm

x, y = load_iris(return_X_y=1)
gmm = Gmm(max_iter=50, n_components=3)
gmm.fit(x)
y_pre = gmm.predict(x)
print(y_pre)
print(y)
