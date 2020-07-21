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

    def forward(self, x_input):
        raise NotImplementedError

    def backward(self, w, grad):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, activation, units):
        self.activation = super().ACTIVATION_MAP[activation]()
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


class SimpleRNN(Layer):
    def __init__(self, hidden_activation, output_activation, max_length, features, hiddenDimension, outputsDimension):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_activations = None
        self.output_activations = None
        self.hiddenDimension = hiddenDimension
        self.outputsDimension = outputsDimension
        self.wa = np.random.normal(size=[hiddenDimension + features, hiddenDimension]) * 0.01
        self.dwa = np.zeros([hiddenDimension + features, hiddenDimension])
        self.wy = np.random.normal(size=[hiddenDimension, outputsDimension]) * 0.01
        self.dwy = np.zeros([hiddenDimension, outputsDimension])
        self.features = features
        self.max_length = max_length
        self.ba = np.zeros(hiddenDimension)
        self.dba = np.zeros(hiddenDimension)
        self.by = np.zeros(outputsDimension)
        self.dby = np.zeros(hiddenDimension)
        self.hidden_vectors = []
        self.output_vectors = []
        self.shape = None
        self._input = None
        self.zy = []

    def padding(self, x_input):
        res = np.zeros([len(x_input), self.max_length, len(x_input[0][0])])
        for i in range(len(x_input)):
            if len(x_input[i]) < self.max_length:
                res[i] += np.concatenate([
                    np.array(x_input[i]), np.zeros([self.max_length - len(x_input[i]), len(x_input[0][0])])], axis=0)
            else:
                res[i] += x_input[i][:self.max_length]
        return res

    def forward(self, x_input: np.ndarray):
        if self.hidden_activations is None:
            x_input = self.padding(x_input)
            self.shape = x_input.shape
            self._input = x_input
            self.hidden_activations = [super().ACTIVATION_MAP[self.hidden_activation]()] * self.shape[1]
            self.output_activations = [super().ACTIVATION_MAP[self.output_activation]()] * self.shape[1]
        self.hidden_vectors = []
        self.output_vectors = []
        self.zy = []
        alpha = np.zeros([self.shape[0], self.hiddenDimension])
        self.hidden_vectors.append(alpha)
        for i in range(self.shape[1]):
            x_concat = np.concatenate([alpha, x_input[:, i, :].reshape(self.shape[0], self.shape[2])], axis=1)
            za = x_concat @ self.wa + self.ba
            alpha = self.hidden_activations[i].forward(za)
            self.hidden_vectors.append(alpha)
            zy = alpha @ self.wy + self.by
            self.zy.append(zy)
            yt = self.output_activations[i].forward(zy)
            self.output_vectors.append(yt)

        return np.array(self.output_vectors).transpose([1, 0, 2])

    def backward(self, grad, w=-1):
        if isinstance(w, int) and w == -1:
            grad = grad
        else:
            grad = grad @ w

        dwy = np.zeros(self.wy.shape)
        dby = np.zeros(self.by.shape)
        dwa = np.zeros(self.wa.shape)
        dba = np.zeros(self.ba.shape)

        grad_a_next = 1
        for i in reversed(range(self.shape[1])):
            _grad = grad[:, i, :].reshape(grad.shape[0], grad.shape[2])
            # 首先更新输出层的w和b，这个和普通全连接层一样
            dzy = _grad * self.output_activations[i].backward()
            dwy += self.hidden_vectors[i].T @ dzy / self.shape[0]
            dby += np.mean(dzy, axis=0)

            # 然后更新dza
            dza = self.hidden_activations[i].backward() * (grad_a_next + dzy @ self.wy.T)

            dwa += np.concatenate(
                [self.hidden_vectors[i - 1], self._input[:, i, :].reshape(self.shape[0], self.shape[2])],
                axis=1).T @ dza
            dba += np.mean(dzy, axis=0)

            grad_a_next = dza
        self.dwy = dwy / self.shape[0]
        self.dby = dby / self.shape[0]
        self.dwa = dwa / self.shape[0]
        self.dba = dba / self.shape[0]

        # 暂时未实现rnn前接rnn的反向传播
        return -1, -1
