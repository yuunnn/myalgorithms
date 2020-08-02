import cupy as np
import numpy
from abc import ABC


class Loss(ABC):
    def loss(self, y_true, y_pre):
        raise NotImplementedError

    def grad(self, y_true, y_pre):
        raise NotImplementedError


class Mse(Loss):

    def loss(self, y_true, y_pre):
        return np.mean((y_true - y_pre) ** 2)

    def loss_numpy(self, y_true, y_pre):
        return numpy.mean((y_true - y_pre) ** 2)

    def grad(self, y_true, y_pre):
        return y_pre - y_true

    def clean(self):
        self.loss = self.loss_numpy()


class Mse2d(Loss):

    def loss(self, y_true, y_pre):
        return np.mean((y_true.reshape(-1) - y_pre.reshape(-1)) ** 2)

    def loss_numpy(self, y_true, y_pre):
        return numpy.mean((y_true.reshape(-1) - y_pre.reshape(-1)) ** 2)

    def grad(self, y_true, y_pre):
        return y_pre - y_true

    def clean(self):
        self.loss = self.loss_numpy()


class Crossentropy_with_softmax(Loss):
    """y_true should be in (0.1.2...n)"""

    def __init__(self, classes):
        self.classes = classes

    def to_onehot(self, y):
        y_true_one_hot = np.array(
            [[1 if _class == m else 0 for _class in range(self.classes)] for m in y])
        return y_true_one_hot

    def to_onehot_numpy(self, y):
        y_true_one_hot = numpy.array(
            [[1 if _class == m else 0 for _class in range(self.classes)] for m in y])
        return y_true_one_hot

    def loss(self, y_true, y_pre):
        m = len(y_true)
        y_true_one_hot = self.to_onehot(y_true)
        return np.sum(-np.log(y_pre) * y_true_one_hot) / m

    def loss_numpy(self, y_true, y_pre):
        m = len(y_true)
        y_true_one_hot = self.to_onehot_numpy(y_true)
        return numpy.sum(-numpy.log(y_pre) * y_true_one_hot) / m

    def grad(self, y_true, y_pre):
        y_true_one_hot = self.to_onehot(y_true)
        return y_pre - y_true_one_hot

    def grad_numy(self, y_true, y_pre):
        y_true_one_hot = self.to_onehot_numpy(y_true)
        return y_pre - y_true_one_hot

    def clean(self):
        self.to_onehot = self.to_onehot_numpy
        self.loss = self.loss_numpy
        self.grad = self.grad_numy


LOSS_MAP = {'Crossentropy_with_softmax': Crossentropy_with_softmax, "Mse": Mse, "mse2d": Mse2d}
