import numpy as np
from .cart import CartRegressor
from .utils import sigmoid


class GbdtRegressor:
    def __init__(self, learning_rate=0.01, max_iter=100, loss='mse', max_depth=3, min_leave_samples=3):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss
        self.max_depth = max_depth
        self.min_leave_samples = min_leave_samples
        self._trees = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.loss == 'mse':
            _f = np.mean(y)
            _r = y - _f
            self._trees.append(np.mean(y))
        elif self.loss == 'mae':
            _f = np.median(y)
            _r = (y - _f > 0) * 2 - 1
            self._trees.append(np.median(y))
        else:
            raise ValueError('loss should be mse or mae')

        for _iter in range(self.max_iter):
            _cart = CartRegressor(max_depth=self.max_depth, min_leave_samples=self.min_leave_samples)
            _cart.fit(x, _r)
            _predict = _cart.predict(x)
            self._trees.append(_cart)
            if self.loss == 'mse':
                _f += _predict * self.learning_rate
                _r = y - _f
            elif self.loss == 'mae':
                _f += _predict * self.learning_rate
                _r = (y - _f > 0) * 2 - 1
            else:
                raise ValueError

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._trees[0] + \
               np.sum([self._trees[i].predict(x) * self.learning_rate for i in range(1, len(self._trees))], axis=0)


class GbdtClassifier:
    def __init__(self, learning_rate=0.01, max_iter=100, loss='deviance', max_depth=3, min_leave_samples=3):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss
        self.max_depth = max_depth
        self.min_leave_samples = min_leave_samples
        self._trees = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        assert len(set(y)) == 2, "暂只支持二分类"
        if self.loss == 'deviance':
            _f = np.mean(y)
            _r = y - sigmoid(_f)
            self._trees.append(_f)
        else:
            raise ValueError('loss should be mse or deviance')

        for _iter in range(self.max_iter):
            _cart = CartRegressor(max_depth=self.max_depth, min_leave_samples=self.min_leave_samples)
            _cart.fit(x, _r)
            _predict = _cart.predict(x)
            self._trees.append(_cart)
            if self.loss == 'deviance':
                _f += _predict * self.learning_rate
                _r = y - sigmoid(_f)
            else:
                raise ValueError("loss should be deviance")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._trees[0] + \
               np.sum([self._trees[i].predict(x) * self.learning_rate for i in range(1, len(self._trees))], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        _proba = self.predict_proba(x)
        return np.array([1 if i >= 0.5 else 0 for i in _proba])
