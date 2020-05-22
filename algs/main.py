from algs.ml_algorithms.cart import Cart
from algs.ml_algorithms.utils import compute_mse
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd
import time

from sklearn.model_selection import train_test_split

# x, y = load_iris(return_X_y=1)
# x = pd.DataFrame(x, columns=['f1', 'f2', 'f3', 'f4'])
x, y = load_boston(return_X_y=1)
# x = pd.DataFrame(x, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13'])
# y = pd.Series(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

cart = Cart(max_depth=10, min_leave_samples=3)
a = time.time()
cart.fit(x_train, y_train)
b  = time.time()
print(b-a)
print(cart._tree.travel(method='pre'))
y_predict = cart.predict(x_test)

# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_predict = lr.predict(x_test)

print(compute_mse(y_predict, y_test))
