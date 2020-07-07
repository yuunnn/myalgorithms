import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from algs.ml_algorithms.cart import CartRegressor
from algs.utils import compute_mse

x, y = load_boston(return_X_y=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

cart = CartRegressor(max_depth=10, min_leave_samples=3)
cart.fit(x_train, y_train)

y_predict = cart.predict(x_test.astype(np.float32))
print(compute_mse(y_predict, y_test))
