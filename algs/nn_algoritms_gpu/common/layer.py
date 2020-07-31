import cupy as np
from abc import ABC
from functools import reduce
from cupy.lib.stride_tricks import as_strided


class Activation(ABC):
    def __init__(self):
        self._input = None

    def forward(self, x_input):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input
        return np.maximum(x_input, 0)

    def backward(self):
        return (self._input > 0) * 1

    def clean(self):
        if self._input is not None:
            self._input = None


class LeakyRelu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input
        right = np.maximum(x_input, 0)
        left = np.min(x_input * 0.001, 0)
        return right + left

    def backward(self):
        right = (self._input >= 0) * 1
        left = (self._input < 0) * 1 * 0.001
        return right + left

    def clean(self):
        if self._input is not None:
            self._input = None


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.g = None

    def forward(self, x_input):
        self.g = 1 / (1 + np.exp(-x_input))
        self._input = x_input
        return self.g

    def backward(self):
        return self.g * (1 - self.g)

    def clean(self):
        if self._input is not None:
            self._input = None
            self.g = None


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.g = None

    def forward(self, x_input):
        self._input = x_input
        self.g = np.tanh(x_input)
        return self.g

    def backward(self):
        return 1 - self.g ** 2

    def clean(self):
        if self._input is not None:
            self._input = None
            self.g = None


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        self._input = x_input.shape
        return x_input

    def backward(self):
        return np.ones(self._input)

    def clean(self):
        if self._input is not None:
            self._input = None


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self._input = None
        self.g = None

    def forward(self, x_input):
        exps = np.exp(x_input)
        self._input = x_input
        self.g = exps / np.sum(exps, axis=1, keepdims=True)
        return self.g

    def backward(self):
        # softmax暂时只能和Crossentropy_with_softmax一起用，并且不能作为普通的激活函数使用
        return 1

    def clean(self):
        if self._input is not None:
            self._input = None
            self.g = None


class Layer(ABC):
    ACTIVATION_MAP = {'relu': Relu, 'sigmoid': Sigmoid, 'linear': Linear, 'softmax': Softmax, 'tanh': Tanh,
                      'leakyrelu': LeakyRelu}

    def forward(self, x_input):
        raise NotImplementedError

    def backward(self, w, grad):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, activation, units):
        self.activation = super().ACTIVATION_MAP[activation]()
        self.b = np.zeros(units).reshape(-1)
        self.units = units
        self.w = None
        self.m = None
        self.grad = None
        self._input = None
        self.dz = None
        self.dw = None
        self.db = None

    def forward(self, x_input):
        self._input = x_input
        if self.w is None:
            self.w = np.random.normal(size=[self.units, x_input.shape[1]]) * 0.01
            self.m = x_input.shape[0]

        return self.activation.forward(x_input @ self.w.T + self.b)

    def backward(self, w, grad):
        dg = self.activation.backward()
        if isinstance(w, int) and w == -1:
            self.dz = grad * dg
        else:
            self.dz = grad @ w * dg
        self.dw = []
        self.dw = self.dz.T @ self._input / self.m
        self.db = np.mean(self.dz, axis=0)
        return self.w, self.dz

    def clean(self):
        if self._input is not None:
            self.m = None
            self._input = None
            self.dz = None
            self.dw = None
            self.db = None

        if hasattr(self, "vdw"):
            del self.vdw
        if hasattr(self, "vdb"):
            del self.vdb

        self.activation.clean()

        self.w = np.asnumpy(self.w)
        self.b = np.asnumpy(self.b)


class SimpleRNN(Layer):
    def __init__(self, hidden_activation, output_activation, max_length, features, hiddenDimension, outputsDimension):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_activations = None
        self.output_activations = None
        self.hiddenDimension = hiddenDimension
        self.outputsDimension = outputsDimension
        self.wa = np.random.normal(size=[hiddenDimension + features, hiddenDimension]) * 0.01
        self.dwa = np.zeros([hiddenDimension + features, hiddenDimension])
        self.wy = np.random.normal(size=[hiddenDimension, outputsDimension]) * 0.01
        self.dwy = np.zeros([hiddenDimension, outputsDimension])
        self.features = features
        self.max_length = max_length
        self.ba = np.zeros(hiddenDimension)
        self.dba = np.zeros(hiddenDimension)
        self.by = np.zeros(outputsDimension)
        self.dby = np.zeros(hiddenDimension)
        self.hidden_vectors = []
        self.output_vectors = []
        self.shape = None
        self._input = None
        self.zy = []

    def padding(self, x_input):
        res = np.zeros([len(x_input), self.max_length, len(x_input[0][0])])
        for i in range(len(x_input)):
            if len(x_input[i]) < self.max_length:
                res[i] += np.concatenate([
                    np.array(x_input[i]), np.zeros([self.max_length - len(x_input[i]), len(x_input[0][0])])], axis=0)
            else:
                res[i] += x_input[i][:self.max_length]
        return res

    def forward(self, x_input: np.ndarray):
        if self.hidden_activations is None:
            x_input = self.padding(x_input)
            self.shape = x_input.shape
            self._input = x_input
            self.hidden_activations = [super().ACTIVATION_MAP[self.hidden_activation]()] * self.shape[1]
            self.output_activations = [super().ACTIVATION_MAP[self.output_activation]()] * self.shape[1]
        self.hidden_vectors = []
        self.output_vectors = []
        self.zy = []
        alpha = np.zeros([x_input.shape[0], self.hiddenDimension])
        self.hidden_vectors.append(alpha)
        for i in range(self.shape[1]):
            x_concat = np.concatenate([alpha, x_input[:, i, :].reshape(x_input.shape[0], self.shape[2])], axis=1)
            za = x_concat @ self.wa + self.ba
            alpha = self.hidden_activations[i].forward(za)
            self.hidden_vectors.append(alpha)
            zy = alpha @ self.wy + self.by
            self.zy.append(zy)
            yt = self.output_activations[i].forward(zy)
            self.output_vectors.append(yt)

        return np.array(self.output_vectors).transpose([1, 0, 2])

    def backward(self, grad, w=-1):
        if isinstance(w, int) and w == -1:
            grad = grad
        else:
            grad = grad @ w

        dwy = np.zeros(self.wy.shape)
        dby = np.zeros(self.by.shape)
        dwa = np.zeros(self.wa.shape)
        dba = np.zeros(self.ba.shape)

        grad_a_next = 0
        for i in reversed(range(self.shape[1])):
            _grad = grad[:, i, :].reshape(grad.shape[0], grad.shape[2])
            # 首先更新输出层的w和b，这个和普通全连接层一样
            dzy = _grad * self.output_activations[i].backward()
            dwy += self.hidden_vectors[i].T @ dzy
            dby += np.sum(dzy)

            # 然后更新dza
            dza = self.hidden_activations[i].backward() * (grad_a_next + dzy @ self.wy.T)

            dwa += np.concatenate(
                [self.hidden_vectors[i - 1], self._input[:, i, :].reshape(self.shape[0], self.shape[2])],
                axis=1).T @ dza
            dba += np.sum(dzy)

            grad_a_next = dza @ self.wa.T[:, :self.hiddenDimension]

        self.dwy = dwy / self.shape[0] / self.shape[1]
        self.dby = dby / self.shape[0] / self.shape[1]
        self.dwa = dwa / self.shape[0] / self.shape[1]
        self.dba = dba / self.shape[0] / self.shape[1]

        # 暂时未实现rnn前接rnn的反向传播
        return -1, -1

    def clean(self):
        if self.hidden_vectors is not []:
            self.hidden_vectors = []
            self.output_vectors = []
            self.zy = []

        for ac in self.hidden_activations:
            ac.clean()

        for ac in self.output_activations:
            ac.clean()

        self.wa = np.asnumpy(self.wa)
        self.wy = np.asnumpy(self.wy)
        self.ba = np.asnumpy(self.ba)
        self.by = np.asnumpy(self.by)

        self.dwa = None
        self.dwy = None
        self.dba = None
        self.dby = None

        if hasattr(self, "vdwa"):
            del self.vdwa
            del self.vdwy
            del self.vdba
            del self.vdby


class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape - 1
        self.shape = None

    def forward(self, x_input):
        self.shape = x_input.shape
        return x_input.reshape(x_input.shape[0], reduce(lambda x, y: x * y, x_input.shape[1:]))

    def backward(self, grad, w=None):
        return -1, (grad @ w).reshape(self.shape)

    def clean(self):
        pass


class ZeroPadding2d(Layer):
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def forward(self, x_input):
        return np.pad(x_input, ((0, 0), (0, 0), (self.l, self.r), (self.l, self.r)), 'constant', constant_values=0)

    def clip(self, grad):
        return grad[:, :, self.l:-self.r, self.l:-self.r]

    def backward(self, grad, w=-1):
        if w == -1:
            return -1, self.clip(grad)
        return -1, self.clip(grad @ w)

    def clean(self):
        pass


class Conv2d(Layer):

    def __init__(self, activation, units, kernel_size, strides, padding='same'):
        assert padding in ['same', 'valid']
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1], ValueError("暂时仅支持正方形卷积核")
        self.activation = super().ACTIVATION_MAP[activation]()
        self.padding = padding
        self.b = np.zeros(units).reshape(-1)
        self.units = units
        self.w = None
        self.m = None
        self.grad = None
        self._input = None
        self.dz = None
        self.dw = None
        self.db = None
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_layer = None
        self.padding_layer_bp = None
        self.shape = None
        self.input_split = None

    @staticmethod
    def split_by_strides(X, kh, kw, s):
        """
         reference 1,一种卷积算法的高效实现 :https://zhuanlan.zhihu.com/p/64933417
         (forward 借鉴了这个，backward是自己推的）

        :param X: 原矩阵
        :param kh: 卷积核h
        :param kw: 卷积核w
        :param s: 步长
        :return:
        """
        N, C, H, W = X.shape
        oh = (H - kh) // s + 1
        ow = (W - kw) // s + 1
        strides = (*X.strides[:-2], X.strides[-2] * s, X.strides[-1] * s, *X.strides[-2:])
        A = as_strided(X, shape=(N, C, oh, ow, kh, kw), strides=strides)
        return A

    def forward(self, x_input):
        """
        :param x_input:  n,c,w.h
        """
        assert x_input.dtype == 'float16', ValueError("输入图片请归一化到0-1，并且请astype到float16")
        assert len(x_input.shape) == 4, ValueError("输入数据必须是4维的，分别是NCWH")
        assert x_input.shape[2] == x_input.shape[3], ValueError("输入图片必须是正方形，维度分别是NCWH")

        # 首先进行padding
        if self.padding == 'same':
            padding_size = (x_input.shape[2] - 1) * self.strides + self.kernel_size[0] - x_input.shape[2]
            if padding_size % 2 == 0:
                self.padding_layer = ZeroPadding2d(padding_size // 2, padding_size // 2)
            else:
                self.padding_layer = ZeroPadding2d(padding_size // 2, padding_size // 2 + 1)
            x_input = self.padding_layer.forward(x_input)

        n, c, w, h = x_input.shape
        self.shape = x_input.shape
        self.m = n
        if self.w is None:
            self.w = np.random.normal(size=[self.units, c, self.kernel_size[0], self.kernel_size[1]]) * 0.01

        # 然后再按照dot的窗口进行split
        x_input = self.split_by_strides(x_input, kh=self.kernel_size[0], kw=self.kernel_size[1], s=self.strides)
        self.input_split = x_input

        return self.activation.forward(
            np.tensordot(x_input, self.w, axes=[(1, 4, 5), (1, 2, 3)]).transpose([0, 3, 1, 2]) +
            self.b.reshape(-1, 1, 1)).astype('float16')

    def backward(self, w, grad):

        # 这部分的推导可以看 /resource下的cnn_bp.md
        if w != -1:
            grad = grad @ w

        self.dz = grad * self.activation.backward()
        self.dw = np.tensordot(self.dz, self.input_split, axes=[(0, 2, 3), (0, 2, 3)]) / self.m
        self.db = np.mean(self.dz, axis=(0, 2, 3))

        pad_diff = 2 * (self.shape[2] - self.dz.shape[2]) * self.strides
        self.padding_layer_bp = ZeroPadding2d(pad_diff // 2, pad_diff // 2)
        self.dz = self.padding_layer_bp.forward(self.dz)
        self.dz = self.split_by_strides(self.dz, kh=self.kernel_size[0], kw=self.kernel_size[1], s=self.strides)
        # 翻转180度
        self.dz = np.flip(self.dz, axis=4)
        self.dz = np.flip(self.dz, axis=4)
        # 这里炫个技，其实和算dw的tensor dot一样的，但是cupy不支持einsum
        # grad = np.einsum('mcab,nmwhab->ncwh', self.w, self.dz)
        grad = np.tensordot(self.w, self.dz, axes=[(0, 2, 3), (1, 4, 5)]).transpose([1, 0, 2, 3])
        if self.padding_layer is not None:
            return self.padding_layer.backward(w=-1, grad=grad)
        return -1, grad

    def clean(self):
        if self.input_split is not None:
            self.input_split = None
            self.dz = None
            self.dw = None
            self.db = None

        if hasattr(self, "vdw"):
            del self.vdw
        if hasattr(self, "vdb"):
            del self.vdb

        self.activation.clean()

        self.w = np.asnumpy(self.w)
        self.b = np.asnumpy(self.b)

