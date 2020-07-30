from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense
from algs.nn_algoritms.common.utils import save_model, load_model
import numpy as np

x = np.array([[0, 1], [1, 1], [0, 0], [0, 1]])
y = np.array([1, 0, 0, 1])

model = Model(lr=0.02, epoch=10000, loss="Crossentropy_with_softmax", classes=2)
model.add(Dense(activation='leakyrelu', units=50))
model.add(Dense(activation='tanh', units=25))
model.add(Dense(activation='softmax', units=2))
model.fit(x, y)
print(model.predict(x))

import matplotlib.pyplot as plt

plt.plot(model.history['loss'])

save_model(model, "./test.m")
model = load_model("./test.m")