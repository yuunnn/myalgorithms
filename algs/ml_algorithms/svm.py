import numpy as np


class Svm:

    def __init__(self, c=1, max_iter=10000):
        self.c = c
        self.max_iter = max_iter
        self.alpha = None
        self.bias = None
        self.support_vector = None
        self.support_vector_sign = None

    def smo(self, x: np.ndarray, y: np.ndarray) -> None:
        c = self.c
        b = 0.1
        m = x.shape[0]
        #alpha = [np.random.random() * c for i in range(m)]
        alpha = [0] * m

        def k(x1, x2):
            return np.dot(x1, x2)

        # init Ei
        e = [0] * m
        e = [np.sum([alpha[j] * y[j] * k(x[j], x[iter_]) for j in range(m)]) + b - y[iter_] for iter_ in range(m)]

        for iter_ in range(self.max_iter):
            # check kkt
            for a1 in range(m):
                if alpha[a1] == 0:
                    if y[a1] * (e[a1] + y[a1]) >= 1:
                        continue
                elif 0 < alpha[a1] < c:
                    if y[a1] * (e[a1] + y[a1]) == 1:
                        continue
                else:
                    if y[a1] * (e[a1] + y[a1]) <= 1:
                        continue

                e1 = e[a1]

                if e1 >= 0:
                    a2 = np.argmin(e)
                    e2 = e[a2]
                else:
                    a2 = np.argmax(e)
                    e2 = e[a2]
                # print('#######')
                # print(y[a1])
                # print(y[a2])
                # print('#######')

                if y[a1] != y[a2]:
                    L = max(0, alpha[a2] - alpha[a1])
                    H = min(c, c + alpha[a2] - alpha[a1])
                else:
                    L = max(0, alpha[a2] + alpha[a1] - c)
                    H = min(c, alpha[a2] + alpha[a1])

                eta = np.sum((x[a1] - x[a2]) ** 2)
                alpha_a2_old = alpha[a2]
                alpha_a1_old = alpha[a1]
                a2_unc = alpha_a2_old + y[a2] * (e1 - e2) / eta

                alpha[a2] = np.clip(a2_unc, L, H)
                alpha[a1] = alpha_a1_old + y[a1] * y[a2] * (alpha_a2_old - alpha[a2])
                b1 = -e1 - y[a1] * k(x[a1], x[a1]) * (alpha[1] - alpha_a1_old) - \
                     y[a2] * k(x[a2], x[a1]) * (alpha[a2] - alpha_a2_old) + b
                b2 = -e2 - y[a1] * k(x[a1], x[a2]) * (alpha[1] - alpha_a1_old) - \
                     y[a2] * k(x[a2], x[a2]) * (alpha[a2] - alpha_a2_old) + b
                b = 0.5 * (b1 + b2)

                e[a1] = np.sum([alpha[j] * y[j] * k(x[j], x[a1]) for j in range(m)]) + b - y[a1]
                e[a2] = np.sum([alpha[j] * y[j] * k(x[j], x[a2]) for j in range(m)]) + b - y[a2]

        none_zero_index = [i for i in range(len(alpha)) if abs(alpha[i]) >= 1e-4]
        self.support_vector = x[none_zero_index]
        self.alpha = np.array(alpha)[none_zero_index]
        self.support_vector_sign = y[none_zero_index]
        self.bias = b

        return

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        y = np.array([1 if i > 0 else -1 for i in y])
        self.smo(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        def sign(y):
            return 1 if y > 0 else -1

        def k(x1, x2):
            return np.dot(x1, x2)

        res = []
        for i in range(x.shape[0]):
            res.append(sign(
                np.sum(
                    [self.alpha[j] * self.support_vector_sign[j] * k(x[i], self.support_vector[j])
                     for j in range(len(self.alpha))] + self.bias
                )
            ))
        return np.array(res)
