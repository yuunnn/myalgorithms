import numpy as np
from abc import ABC


class Loss(ABC):
    def loss(self, y_true, y_pre):
        raise NotImplementedError

    def grad(self, y_true, y_pre):
        raise NotImplementedError


class Mse(Loss):

    def loss(self, y_true, y_pre):
        return np.mean((y_true - y_pre.reshape(-1)) ** 2)

    def grad(self, y_true, y_pre):
        return (y_pre.reshape(-1) - y_true).reshape(-1, 1)


class Crossentropy_with_softmax(Loss):
    """y_true should be in (0.1.2...n)"""

    def __init__(self):
        self.y_true_one_hot = None

    def loss(self, y_true, y_pre):
        m = len(y_true)
        if self.y_true_one_hot is None:
            classes = len(set(y_true))
            self.y_true_one_hot = np.array(
                [[1 if _class == m else 0 for _class in range(classes)] for m in y_true])

        return np.sum(-np.log(y_pre) * self.y_true_one_hot) / m

    def grad(self, y_true, y_pre):
        return y_pre - self.y_true_one_hot


LOSS_MAP = {'Crossentropy_with_softmax': Crossentropy_with_softmax, "Mse": Mse}
