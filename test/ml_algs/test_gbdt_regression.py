from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from algs.ml_algorithms.gbdt import GbdtRegressor
from algs.utils import compute_mse, compute_mae

x, y = load_boston(return_X_y=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

gbdt = GbdtRegressor(max_depth=5, min_leave_samples=5, max_iter=200, loss='mse', learning_rate=0.1)
gbdt.fit(x_train, y_train)
y_predict = gbdt.predict(x_test)
print(compute_mse(y_predict, y_test))

gbdt = GbdtRegressor(max_depth=5, min_leave_samples=5, max_iter=200, loss='mae', learning_rate=0.1)
gbdt.fit(x_train, y_train)
y_predict = gbdt.predict(x_test)
print(compute_mae(y_predict, y_test))
