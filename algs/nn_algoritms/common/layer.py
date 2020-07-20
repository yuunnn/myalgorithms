import numpy as np
from abc import ABC


class Activation(ABC):
    def __init__(self):
        self._input = None

    def forward(self, x_input):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input
        return np.maximum(x_input, 0)

    def backward(self):
        return (self._input > 0) * 1


class LeakyRelu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input
        right = np.maximum(x_input, 0)
        left = np.min(x_input * 0.001, 0)
        return right + left

    def backward(self):
        right = (self._input >= 0) * 1
        left = (self._input < 0) * 1 * 0.001
        return right + left


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.g = None

    def forward(self, x_input):
        self.g = 1 / (1 + np.exp(-x_input))
        self._input = x_input
        return self.g

    def backward(self):
        return self.g * (1 - self.g)


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.g = None

    def forward(self, x_input):
        self._input = x_input
        self.g = np.tanh(x_input)
        return self.g

    def backward(self):
        return 1 - self.g ** 2


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input.shape
        return x_input

    def backward(self):
        return np.ones(self._input)


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self._input = None
        self.g = None

    def forward(self, x_input):
        exps = np.exp(x_input)
        self._input = x_input
        self.g = exps / np.sum(exps, axis=1, keepdims=True)
        return self.g

    def backward(self):
        # softmax暂时只能和Crossentropy_with_softmax一起用，并且不能作为普通的激活函数使用
        return 1


class Layer(ABC):
    ACTIVATION_MAP = {'relu': Relu, 'sigmoid': Sigmoid, 'linear': Linear, 'softmax': Softmax, 'tanh': Tanh,
                      'leakyrelu': LeakyRelu}

    def __init__(self, activation, units):
        self.activation = self.ACTIVATION_MAP[activation]()
        self.b = np.zeros(units).reshape(-1)
        self.units = units
        self.w = None
        self.m = None
        self.grad = None
        self._input = None
        self.dz = None
        self.dw = None
        self.db = None

    def forward(self, x_input):
        raise NotImplementedError

    def backward(self, w, grad):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, activation, units):
        super().__init__(activation=activation, units=units)

    def forward(self, x_input):
        self._input = x_input
        if self.w is None:
            self.w = np.random.normal(size=[self.units, x_input.shape[1]]) * 0.01
            self.m = x_input.shape[0]

        return self.activation.forward(x_input @ self.w.T + self.b)

    def backward(self, w, grad):
        dg = self.activation.backward()
        if isinstance(w, int) and w == -1:
            self.dz = grad * dg
        else:
            self.dz = grad @ w * dg
        self.dw = []
        self.dw = self.dz.T @ self._input / self.m
        self.db = np.mean(self.dz, axis=0)
        return self.w, self.dz
