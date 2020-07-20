from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from algs.utils import compute_mse

x, y = load_boston(return_X_y=1)
x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

model = Model(lr=0.01, max_iter=1500, loss="Mse")
model.add(Dense(activation='relu', units=64))
model.add(Dense(activation='tanh', units=32))
model.add(Dense(activation='linear', units=1))
model.fit(x_train, y_train)

y_predict = model.predict(x_test).reshape(-1)
print(compute_mse(y_predict, y_test))

import matplotlib.pyplot as plt
plt.plot(model.history['loss'])