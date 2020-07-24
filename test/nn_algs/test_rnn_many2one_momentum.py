"""
many to one mode with momentum for "test_rnn_sin"
"""
from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense, SimpleRNN, Flatten

import numpy as np

data = np.linspace(-100, 100, 2000)
data = np.sin(data)

x = []
y = []
for i in range(1000):
    x.append(data[i:i + 50])
    y.append(data[i + 51])

x = np.array(x).reshape(1000, 50, 1)
y = np.array(y).reshape(1000, 1)

model = Model(lr=0.1, epoch=2500, loss="Mse", optimizer='sgd_with_momentum', decay=0.9999,
              early_stop=True, tol=2e-4, momentum_beta=0.9, batch_size=32, shuffle=1)
model.add(SimpleRNN(hidden_activation='tanh',
                    output_activation='tanh',
                    max_length=50,
                    features=1,
                    hiddenDimension=20,
                    outputsDimension=1))
model.add(Flatten(input_shape=3))
# many to one的时候Flatten层后面一层的单元数量必须和rnn层时间步长一致
model.add(Dense(activation='linear', units=50))
model.add(Dense(activation='linear', units=1))

model.fit(x, y, watch_loss=True)

x_test = []
for i in range(1000, 1500):
    x_test.append(data[i:i + 50])
x_test = np.array(x_test).reshape(500, 50, 1)
y_pre = model.predict(x_test)

import matplotlib.pyplot as plt

plt.plot(y_pre)
plt.plot(data[1050:1550])
