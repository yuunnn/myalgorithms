import numpy as np
from abc import ABC


class Loss(ABC):
    def loss(self, y_ture, y_pred):
        raise NotImplementedError

    def grad(self, y_ture, y_pred):
        raise NotImplementedError


class Mse(Loss):

    def loss(self, y_true, y_pre):
        return np.sum((y_true - y_pre) ** 2)

    def grad(self, y_true, y_pre):
        return (y_pre - y_true).reshape(-1, 1)


class Crossentropy(Loss):
    """y_true should be in (0.1.2...n)"""

    def __init__(self):
        self.y_true_one_hot = None

    def loss(self, y_true, y_pre):
        m = len(y_true)
        if self.y_true_one_hot is None:
            classes = len(set(y_true))
            self.y_true_one_hot = np.array(
                [[1 if _class == m else 0 for _class in range(classes)] for m in y_true])[:,:,np.newaxis]

        return np.sum(-np.log(y_pre) * self.y_true_one_hot) / m

    def grad(self, y_true, y_pre):
        return np.mean(-self.y_true_one_hot / y_pre, axis=0).reshape(-1, 1)


LOSS_MAP = {'Crossentropy': Crossentropy}