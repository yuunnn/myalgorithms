from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense, SimpleRNN

import numpy as np

data = np.linspace(-10, 10, 10000)
data = np.sin(data)

x = []
y = []
for i in range(1000):
    x.append(data[i:i + 100])
    y.append(data[i + 1:i + 101])

x = np.array(x).reshape(1000, 100, 1)
y = np.array(y).reshape(1000, 100, 1)

model = Model(lr=0.002, max_iter=2000, loss="mse2d")
model.add(SimpleRNN(hidden_activation='tanh',
                    output_activation='linear',
                    max_length=1,
                    hiddenDimension=200,
                    outputsDimension=1))
model.fit(x, y, watch_loss=False)

import matplotlib.pyplot as plt
plt.plot(model.history['loss'])
