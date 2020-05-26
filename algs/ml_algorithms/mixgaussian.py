import numpy as np
from functools import reduce

"""
方法来自于《统计学习方法》和吴恩达斯坦福讲义
"""


class Gmm:

    def __init__(self, max_iter=100, n_components=3):
        self.max_iter = max_iter
        self.n_components = n_components
        self.mu = None
        self.covariances = None
        self.gamma = None
        self.alpha = None

    def init(self, x):
        self.mu = x[np.random.choice(range(x.shape[0]), self.n_components)]
        self.covariances = [np.diag([1] * x.shape[1])] * self.n_components
        self.gamma = np.ones([x.shape[0], self.n_components])
        self.alpha = np.ones(self.n_components) / self.n_components

    @staticmethod
    def compute_pdf(x, mu, det_covariances, pinv_covariances):
        _, shape = pinv_covariances.shape
        part1 = 1 / ((2 * np.pi) ** (shape / 2) * det_covariances ** 0.5)
        tmp = (x - mu).reshape(-1, 1)
        part2 = np.exp(-0.5 * reduce(np.matmul, (tmp.T, pinv_covariances, tmp))[0][0])
        return part1 * part2

    def e_step(self, x):
        probability = [[] for j in range(x.shape[0])]
        for k in range(self.n_components):
            det_covariances = np.linalg.det(self.covariances[k])
            pinv_covariances = np.linalg.pinv(self.covariances[k])
            for j in range(x.shape[0]):
                probability[j].append(
                    self.alpha[k] * self.compute_pdf(x[j], self.mu[k], det_covariances, pinv_covariances))
        for k in range(self.n_components):
            for j in range(x.shape[0]):
                self.gamma[j][k] = probability[j][k] / sum(probability[j])

    def m_setp(self, x):
        for k in range(self.n_components):
            self.mu[k] = np.sum([self.gamma[i][k] * x[i] for i in range(x.shape[0])], axis=0) / np.sum(self.gamma[:, k])

            tmp = np.sum([[self.gamma[i][k] * np.matmul((x[i] - self.mu[k]).reshape(-1, 1),
                                                        (x[i] - self.mu[k]).reshape(-1, 1).T)]
                          for i in range(x.shape[0])], axis=0)

            self.covariances[k] = tmp[0] / np.sum(self.gamma[:, k])
            self.covariances[k].flat[::self.n_components + 1] += 1e-04
            self.alpha[k] = np.mean(self.gamma[:, k])

    def fit(self, x: np.ndarray) -> None:
        self.init(x)
        for i in range(self.max_iter):
            self.e_step(x)
            self.m_setp(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        res = self.predict_proba(x)
        res = np.argmax(res, axis=1)
        return np.array(res)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = [[] for j in range(x.shape[0])]
        alpha_probability = [[] for j in range(x.shape[0])]
        for k in range(self.n_components):
            det_covariances = np.linalg.det(self.covariances[k])
            pinv_covariances = np.linalg.pinv(self.covariances[k])
            for j in range(x.shape[0]):
                probability[j].append(
                    self.alpha[k] * self.compute_pdf(x[j], self.mu[k], det_covariances, pinv_covariances))
        for k in range(self.n_components):
            for j in range(x.shape[0]):
                alpha_probability[j].append(probability[j][k] / sum(probability[j]))
        return np.array(alpha_probability)
