from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense
import numpy as np
# x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# x = x[:, :, np.newaxis]
# y = np.array([0, 1, 1, 0])

# x = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
# x = x[:, :, np.newaxis]
# y = np.array([1, 1, 0, 0])

# model = Model(lr=0.01, max_iter=1000, loss="Crossentropy")
# # model.add(Dense(activation='relu', units=50))
# # model.add(Dense(activation='relu', units=50))
# model.add(Dense(activation='sigmoid', units=50))
# model.add(Dense(activation='softmax', units=2))
# model.fit(x, y)
# print(model.predict(x))


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from algs.utils import compute_mse

x, y = load_boston(return_X_y=1)
x = x[:, :, np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

model = Model(lr=0.000005, max_iter=1000, loss="Mse")
# model.add(Dense(activation='relu', units=5))
model.add(Dense(activation='linear', units=1))
model.fit(x_train, y_train)

# y_predict = model.predict(x_test)
# print(compute_mse(y_predict, y_test))
y_predict = model.predict(x_train).reshape(-1)
print(compute_mse(y_predict, y_train))
