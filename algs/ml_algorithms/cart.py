from algs.algorithms.tree import BinaryTree, Node
from collections import namedtuple
import numpy as np


class Cart:
    NodeValue = namedtuple("NodeValue", ['col', 'split', 'res_value'])

    def __init__(self, max_depth, min_leave_samples=3, min_leave_samples_type=1):
        self._tree = BinaryTree()
        self._tree.root = Node(None)
        self.max_depth = max_depth
        self.min_leave_samples = min_leave_samples
        self.min_leave_samples_type = min_leave_samples_type

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:

        def _build(node, depth, _x, _y):

            if depth == 0 or len(_x) <= self.min_leave_samples:
                return

            _index, _col, _split, _loss = None, None, None, +np.inf

            for col in range(x.shape[1]):
                if set(_x[:, col]).__len__() < self.min_leave_samples_type:
                    continue
                sort_indices = np.argsort(_x[:, col])
                _x = _x[sort_indices]
                _y = _y[sort_indices]

                for val_index in range(1, len(_x) - 1):
                    _y1 = _y[val_index:]
                    _y2 = _y[:val_index]
                    _tmp_loss = sum((_y1 - np.mean(_y1)) ** 2) + sum((_y2 - np.mean(_y2)) ** 2)
                    if _tmp_loss < _loss:
                        _col, _split, _loss = col, _x[val_index, col], _tmp_loss

            node.value = self.NodeValue(col=_col, split=_split, res_value=np.mean(_y))
            node.left = Node(None)
            node.right = Node(None)

            _index = _x[:, _col] < _split
            _build(node.left, depth - 1, _x[_index], _y[_index])
            _build(node.right, depth - 1, _x[~_index], _y[~_index])

        _build(self._tree.root, self.max_depth, x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        def _predict(node, _x):
            _col, _split, _res_value = node.value
            if node.left.value is None and node.right.value is None:
                return _res_value

            elif node.left.value is not None and node.right.value is not None:
                if _x[_col] < _split:
                    return _predict(node.left, _x)
                else:
                    return _predict(node.right, _x)
            elif node.left.value is None and node.right.value is not None:
                return _predict(node.right, _x)
            elif node.left.value is not None and node.right.value is None:
                return _predict(node.left, _x)

        res = [_predict(self._tree.root, x[i, :]) for i in range(len(x))]
        return np.array(res)
