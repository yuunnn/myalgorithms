from .layer import *


class Optimizer(ABC):
    def __init__(self, lr, decay):
        self.lr = lr
        self.decay = decay

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def get_decay(self):
        self.lr *= self.decay


class Sgd(Optimizer):
    """实际上是 batch GD 或者mini batch GD"""
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
            elif isinstance(layer, Flatten):
                continue
            else:
                raise NotImplementedError


class Sgd_with_Momentum(Optimizer):
    """实际上是mini batch GD with momentum"""
    def __init__(self, lr, decay, beta):
        super().__init__(lr, decay)
        self.beta = beta

    def compute(self, model):
        for layer in model.layer:
            if isinstance(layer, SimpleRNN):
                if not hasattr(layer, 'vdwy'):
                    layer.vdwy = 0
                    layer.vdwa = 0
                    layer.vdby = 0
                    layer.vdba = 0

                layer.vdwy = self.beta * layer.vdwy + (1 - self.beta) * layer.dwy
                layer.vdwa = self.beta * layer.vdwa + (1 - self.beta) * layer.dwa
                layer.vdby = self.beta * layer.vdby + (1 - self.beta) * layer.dby
                layer.vdba = self.beta * layer.vdba + (1 - self.beta) * layer.dba

                layer.wy -= self.lr * layer.vdwy
                layer.by -= self.lr * layer.vdby
                layer.wa -= self.lr * layer.vdwa
                layer.ba -= self.lr * layer.vdba

            elif isinstance(layer, Dense):
                if not hasattr(layer, 'vdw'):
                    layer.vdw = 0
                    layer.vdb = 0

                layer.vdw = self.beta * layer.vdw + (1 - self.beta) * layer.dw
                layer.vdb = self.beta * layer.vdb + (1 - self.beta) * layer.db

                layer.w -= self.lr * layer.vdw
                layer.b -= self.lr * layer.vdb

            elif isinstance(layer, Flatten):
                continue
            else:
                raise NotImplementedError


OPTIMIZER_MAP = {'sgd': Sgd, 'sgd_with_momentum': Sgd_with_Momentum}
