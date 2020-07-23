import numpy as np
from .layer import *
from .optimizer import OPTIMIZER_MAP
from .loss import LOSS_MAP


class Model:
    def __init__(self, optimizer='sgd', loss='Crossentropy', max_iter=100, lr=0.01, decay=1, early_stop=True,
                 tol=1e-6, momentum_beta=0.9, batch_size=-1, shuffle=False):
        if optimizer == 'sgd_with_momentum':
            self.optimizer = OPTIMIZER_MAP[optimizer](lr, decay, momentum_beta)
        elif optimizer == 'sgd':
            self.optimizer = OPTIMIZER_MAP[optimizer](lr, decay)
        else:
            raise NotImplementedError("optimizer now should be sgd or sgd_with_momentum")
        self.loss = LOSS_MAP[loss]()
        self.layer = []
        self.max_iter = max_iter
        self.history = {'loss': []}
        self.early_stop = early_stop
        self.tol = tol
        self.batch_size = batch_size
        self.shuffle = shuffle

    def add(self, layer):
        self.layer.append(layer)

    def compile(self):
        pass

    def forward(self, x):
        for _layer in self.layer:
            x = _layer.forward(x)
        return x

    def backward(self, grad):
        w = -1
        for _layer in reversed(self.layer):
            w, grad = _layer.backward(w=w, grad=grad)
        return

    def step(self):
        self.optimizer.compute(self)

    def fit(self, x, y, watch_loss=False):
        _x = x[:]
        _y = y[:]
        for _iter in range(self.max_iter):
            if self.shuffle:
                _index = list(range(_x.shape[0]))
                np.random.shuffle(_index)
                _x = _x[_index]
                _y = _y[_index]
            if self.batch_size == -1:
                batch_nums = 1
                self.batch_size = x.shape[0]
            else:
                batch_nums = int(_x.shape[0] / self.batch_size)

            for i in range(batch_nums):
                res = self.forward(_x[i * self.batch_size: (i + 1) * self.batch_size])
                grad = self.loss.grad(_y[i * self.batch_size:(i + 1) * self.batch_size], res)
                self.backward(grad)
                self.step()

            self.optimizer.get_decay()
            res = self.forward(_x)
            loss = self.loss.loss(_y, res)
            self.history['loss'].append(loss)
            if watch_loss:
                print("iter {}:    {}".format(_iter, loss))
            if self.early_stop:
                if loss < self.tol:
                    break

    def predict(self, x):
        return self.forward(x)
