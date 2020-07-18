import numpy as np
from .layer import *
from .optimizer import OPTIMIZER_MAP
from .loss import LOSS_MAP


class Model:
    def __init__(self, optimizer='sgd', loss='Crossentropy', max_iter=100, lr=0.01):
        self.optimizer = OPTIMIZER_MAP[optimizer](lr)
        self.loss = LOSS_MAP[loss]()
        self.layer = []
        self.max_iter = max_iter

    def add(self, layer):
        self.layer.append(layer)

    def compile(self):
        pass

    def forward(self, x):
        tmp = x[:]
        for _layer in self.layer:
            tmp = _layer.forward(tmp)
        return tmp

    def backward(self, grad):
        w = -1
        for _layer in reversed(self.layer):
            w, grad = _layer.backward(w, grad)
        return

    def step(self):
        # a = np.sum(self.layer[0].w)
        self.optimizer.compute(self)
        # b = np.sum(self.layer[0].w)
        # print(b-a)
        # print(self.layer[0].dw[0])

    def fit(self, x, y):
        for _iter in range(self.max_iter):
            res = self.forward(x)
            loss = self.loss.loss(y, res)
            print("iter: {}, loss: {}".format(_iter, loss))
            grad = self.loss.grad(y, res)
            self.backward(grad)
            self.step()

    def predict(self, x):
        return self.forward(x)
