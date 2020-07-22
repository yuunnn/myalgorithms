from abc import ABC
from .layer import *


class Optimizer(ABC):
    def __init__(self, lr, decay):
        self.lr = lr
        self.decay = decay

    def compute(self, *args, **kwargs):
        raise NotImplementedError


class Sgd(Optimizer):
    def __init__(self, lr, decay):
        super().__init__(lr, decay)

    def compute(self, model):
        for layer in model.layer:
            if isinstance(layer, SimpleRNN):
                layer.wy -= self.lr * layer.dwy
                layer.by -= self.lr * layer.dby
                layer.wa -= self.lr * layer.dwa
                layer.ba -= self.lr * layer.dba
            elif isinstance(layer, Dense):
                layer.w -= self.lr * layer.dw
                layer.b -= self.lr * layer.db
            else:
                raise NotImplementedError
        self.lr *= self.decay


OPTIMIZER_MAP = {'sgd': Sgd}
