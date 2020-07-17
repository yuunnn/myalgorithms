from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense
import numpy as np

x = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])
x = x[:, :, np.newaxis]
y = np.array([0, 1, 0, 1])

model = Model(lr=0.2)
model.add(Dense(activation='sigmoid', units=2))
model.add(Dense(activation='sigmoid', units=2))
model.add(Dense(activation='softmax', units=2))
model.fit(x, y)

print(np.argmax(model.predict(x), axis=1))
