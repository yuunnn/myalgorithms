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
        return np.mean((self._input > 0) * 1, axis=0)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.g = None

    def forward(self, x_input):
        self.g = 1 / (1 + np.exp(-x_input))
        self._input = x_input
        return self.g

    def backward(self):
        return np.mean(self.g * (1 - self.g), axis=0)


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input.shape
        return x_input

    def backward(self):
        return np.mean(np.ones(self._input), axis=0)


# class Softmax(Activation):
#     def __init__(self):
#         super().__init__()
#         self._input = None
#
#     def forward(self, x_input):
#         exps = np.exp(x_input)
#         self._input = x_input
#         return exps / np.sum(exps, axis=1).reshape(-1, 1, 1)
#
#     def backward(self):
#         sumexps = np.sum(np.exp(self._input), axis=1).reshape(-1, 1, 1)
#         exps2x = np.exp(self._input * 2)
#         return np.mean((np.exp(self._input) * sumexps - exps2x) / (sumexps ** 2), axis=0)
#
#
# class Softmax(Activation):
#     def __init__(self):
#         super().__init__()
#         self._input = None
#         self.g = None
#
#     def forward(self, x_input):
#         exps = np.exp(x_input)
#         self._input = x_input
#         self.g = exps / np.sum(exps, axis=1).reshape(-1, 1, 1)
#         return self.g
#
#     def backward(self):
#         return np.mean(self.g * (1 - self.g), axis=0)


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self._input = None
        self.g = None

    def forward(self, x_input):
        exps = np.exp(x_input)
        self._input = x_input
        self.g = exps / np.sum(exps, axis=1).reshape(-1, 1, 1)
        return self.g

    def backward(self):
        res = []
        for _matrix in self.g:
            res.append(_matrix - sum(set((_matrix * _matrix.T).flat) - set(np.diag(_matrix * _matrix.T))))
        res = np.array(res)
        return np.mean(res, axis=0)


class Layer(ABC):
    ACTIVATION_MAP = {'relu': Relu, 'sigmoid': Sigmoid, 'softmax': Softmax, 'linear': Linear}

    def __init__(self, activation, units):
        self.activation = activation()
        self.b = np.zeros(units).reshape(-1, 1)
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
        super().__init__(activation=Layer.ACTIVATION_MAP[activation], units=units)

    def forward(self, x_input):
        self._input = x_input
        if self.w is None:
            self.w = np.random.normal(size=[self.units, x_input.shape[1]])[np.newaxis, :, :] * 0.01
            self.m = x_input.shape[0]

        return self.activation.forward(self.w @ x_input + self.b)

    def backward(self, w, grad):
        dg = self.activation.backward()
        if isinstance(w, int) and w == -1:
            self.dz = grad * dg
        else:
            self.dz = (w[0].T @ grad * dg).reshape(-1, 1)
        self.dw = (self.dz @ np.mean(self._input, axis=0, keepdims=True).reshape(1, -1))[np.newaxis, :, :]
        self.db = 1 / self.m * np.sum(self.dz, axis=1, keepdims=True)
        return self.w, self.dz
