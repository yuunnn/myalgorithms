from algs.ml_algorithms.cart import CartRegressor, CartClassifier
from algs.ml_algorithms.mixgaussian import Gmm
from algs.ml_algorithms.utils import compute_mse, compute_logloss, compute_confusion_matrix
from algs.ml_algorithms.svm import Svm
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

# cartregressor
# x, y = load_boston(return_X_y=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

# cart = CartRegressor(max_depth=10, min_leave_samples=3)
# a = time.time()
# cart.fit(x_train, y_train)
# b = time.time()
# print(b - a)

# nodes = cart._tree.travel(method='pre')
# for node in nodes:
#     print(node)
# y_predict = cart.predict(x_test.astype(np.float32))

# print(y_predict)
# print(compute_mse(y_predict, y_test))


# cartclassifier
# x, y = load_breast_cancer(return_X_y=1)
# x, y = load_iris(return_X_y=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
#
# a = time.time()
# cart = CartClassifier(max_depth=5, min_leave_samples=5)
# cart2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
# cart.fit(x_train, y_train)
# cart2.fit(x_train, y_train)
#
# b = time.time()
# print(b - a)
#
# y_predict = cart.predict(x_test)
# # print(y_predict)
# print(compute_confusion_matrix(y_predict, y_test))
# print("##################")
# print(compute_confusion_matrix(cart2.predict(x_test), y_test))

# gmm
# x, y = load_iris(return_X_y=1)
# gmm = Gmm(max_iter=50, n_components=3)
# gmm.fit(x)
# proba = gmm.predict_proba(x)
# y_pre = gmm.predict(x)
# print(y_pre)
# print(y)


# svm
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


x, y = load_breast_cancer(return_X_y=1)
x = standardization(x)


svm = Svm(max_iter=100)
svm.fit(x, y)
print(svm.predict(x))
print(svm.alpha)
