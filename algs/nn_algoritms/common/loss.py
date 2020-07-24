import numpy as np
from abc import ABC


class Loss(ABC):
    def loss(self, y_true, y_pre):
        raise NotImplementedError

    def grad(self, y_true, y_pre):
        raise NotImplementedError


class Mse(Loss):

    def loss(self, y_true, y_pre):
        return np.mean((y_true - y_pre) ** 2)

    def grad(self, y_true, y_pre):
        return y_pre - y_true


class Mse2d(Loss):

    def loss(self, y_true, y_pre):
        return np.mean((y_true.reshape(-1) - y_pre.reshape(-1)) ** 2)

    def grad(self, y_true, y_pre):
        return y_pre - y_true


class Crossentropy_with_softmax(Loss):
    """y_true should be in (0.1.2...n)"""

    def __init__(self):
        self.y_true_one_hot = None

    @staticmethod
    def to_onehot(y):
        classes = len(set(y))
        y_true_one_hot = np.array(
            [[1 if _class == m else 0 for _class in range(classes)] for m in y])
        return y_true_one_hot
        
    def loss(self, y_true, y_pre):
        m = len(y_true)
        y_true_one_hot = self.to_onehot(y_true)
        return np.sum(-np.log(y_pre) * y_true_one_hot) / m

    def grad(self, y_true, y_pre):
        y_true_one_hot = self.to_onehot(y_true)
        return y_pre - y_true_one_hot


LOSS_MAP = {'Crossentropy_with_softmax': Crossentropy_with_softmax, "Mse": Mse, "mse2d": Mse2d}
