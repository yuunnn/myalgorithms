from typing import Optional, Union

from algs.data_structure_algorithms.tree import BinaryTree, Node
import numpy as np


class NodeValue:
    def __init__(self, value):
        self._value = value
        self._col = None
        self._split = None

    def __call__(self):
        return self.value, self.col, self.split

    def __str__(self):
        return 'value ' + str(self.value) + '  col ' + str(self.col) + '  split ' + str(self.split)

    @property
    def value(self):
        return self._value

    @property
    def col(self):
        return self._col

    @col.setter
    def col(self, col):
        self._col = col

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        self._split = split


class CartNode(Node):
    def __init__(self, value: Optional[Union[int, float, NodeValue]], color: Optional[int] = None):
        super().__init__(value, color)


class CartRegressor:

    def __init__(self, max_depth, min_leave_samples=3):
        self._tree = BinaryTree()
        self._tree.root = Node(None)
        self.max_depth = max_depth
        self.min_leave_samples = min_leave_samples

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:

        def _build(node, depth, _x, _y):

            if depth == 0 or len(_x) <= self.min_leave_samples:
                return

            _col, _split, _loss, _split_y1, _split_y2 = None, None, +np.inf, None, None

            for col in range(x.shape[1]):
                sort_indices = np.argsort(_x[:, col])
                _x = _x[sort_indices]
                _y = _y[sort_indices]

                _sum_y1 = 0
                _sum_y2 = np.sum(_y)
                _len_y = len(_y)

                for val_index in range(_len_y - 1):
                    _sum_y1 += _y[val_index]
                    _sum_y2 -= _y[val_index]

                    _tmp_loss = - _sum_y1 ** 2 / (val_index + 1) - _sum_y2 ** 2 / (_len_y - val_index - 1)

                    if _tmp_loss < _loss:
                        _col = col
                        _split = (_x[val_index, col] + _x[val_index + 1, col]) / 2
                        _loss = _tmp_loss
                        _split_y1 = _sum_y1 / (val_index + 1)
                        _split_y2 = _sum_y2 / (_len_y - val_index - 1)

            node.value.col = _col
            node.value.split = _split
            node.left = CartNode(NodeValue(_split_y1))
            node.right = CartNode(NodeValue(_split_y2))

            _index = _x[:, _col] < _split
            _build(node.left, depth - 1, _x[_index], _y[_index])
            _build(node.right, depth - 1, _x[~_index], _y[~_index])

        self._tree.root.value = NodeValue(np.sum(y))
        _build(self._tree.root, self.max_depth, x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        def _predict(node, _x):
            _res_value, _col, _split, = node.value()
            if node.left is None and node.right is None:
                return _res_value
            elif node.left is not None and node.right is not None:
                if _x[_col] < _split:
                    return _predict(node.left, _x)
                else:
                    return _predict(node.right, _x)
            elif node.left is None and node.right is not None:
                return _predict(node.right, _x)
            else:
                return _predict(node.left, _x)

        res = [_predict(self._tree.root, x[i, :]) for i in range(len(x))]
        return np.array(res)


class CartClassifier:

    def __init__(self, max_depth, min_leave_samples=3):
        self._tree = BinaryTree()
        self._tree.root = Node(None)
        self.max_depth = max_depth
        self.min_leave_samples = min_leave_samples

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _y_dict = dict(zip(set(y), [0] * len(set(y))))

        def _build(node, depth, _x, _y):

            if depth == 0 or len(_x) <= self.min_leave_samples:
                return

            _col, _split, _loss, _split_y1, _split_y2 = None, None, +np.inf, None, None

            for col in range(x.shape[1]):
                sort_indices = np.argsort(_x[:, col])
                _x = _x[sort_indices]
                _y = _y[sort_indices]

                _dict_y1 = _y_dict.copy()
                _dict_y2 = _y_dict.copy()
                for k in _y:
                    _dict_y2[k] = _dict_y2.get(k, 0) + 1

                _len_y = len(_y)

                for val_index in range(_len_y - 1):
                    _dict_y1[_y[val_index]] += 1
                    _dict_y2[_y[val_index]] -= 1

                    d1 = val_index + 1
                    d2 = _len_y - val_index - 1
                    _tmp_loss = d1 * (1 - sum([(c_k / d1) ** 2 for c_k in _dict_y1.values()])) \
                                + d2 * (1 - sum([(c_k / d2) ** 2 for c_k in _dict_y2.values()]))

                    if _tmp_loss < _loss:
                        _col = col
                        _split = (_x[val_index, col] + _x[val_index + 1, col]) / 2
                        _loss = _tmp_loss
                        _split_y1 = _dict_y1
                        _split_y2 = _dict_y2

            node.value.col = _col
            node.value.split = _split
            node.left = CartNode(NodeValue(_split_y1))
            node.right = CartNode(NodeValue(_split_y2))

            _index = _x[:, _col] < _split
            _build(node.left, depth - 1, _x[_index], _y[_index])
            _build(node.right, depth - 1, _x[~_index], _y[~_index])

        self._tree.root.value = NodeValue(np.sum(y))
        _build(self._tree.root, self.max_depth, x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        def _predict(node, _x):
            _res_dict, _col, _split, = node.value()

            if node.left is None and node.right is None:
                _tmp = sorted(_res_dict.items(), key=lambda _x: _x[0])
                _tmp = [i[1] for i in _tmp]
                _res = [i / sum(_tmp) for i in _tmp]
                _res = [max(0.0001, min(1 - 0.0001, i)) for i in _res]
                return _res
            elif node.left is not None and node.right is not None:
                if _x[_col] < _split:
                    return _predict(node.left, _x)
                else:
                    return _predict(node.right, _x)
            elif node.left is None and node.right is not None:
                return _predict(node.right, _x)
            else:
                return _predict(node.left, _x)

        res = [_predict(self._tree.root, x[i, :]) for i in range(len(x))]
        return np.array(res)

    def predict(self, x: np.ndarray) -> np.ndarray:
        res = self.predict_proba(x)
        res = [np.argmax(i) for i in res]
        return np.array(res)
