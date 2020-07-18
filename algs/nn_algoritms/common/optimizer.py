from abc import ABC
import numpy as np


class Optimizer(ABC):
    def __init__(self, lr):
        self.lr = lr

    def compute(self, *args, **kwargs):
        raise NotImplementedError


class Sgd(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def compute(self, model):
        for layer in model.layer:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db


OPTIMIZER_MAP = {'sgd': Sgd}
