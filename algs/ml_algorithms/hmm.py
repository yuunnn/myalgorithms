"""
全部公式来源《统计学习方法》
"""

import numpy as np


class Hmm:
    def __init__(self, A=None, B=None, Pi=None, N=None, K=None):
        self.A = A  # 状态转移矩阵
        self.B = B  # 观测概率矩阵
        self.Pi = Pi  # 初始状态概率向量
        self.N = N  # N 为所有样本的所有状态集合,为了便于固定A和Pi，转为一个字典
        self.K = K  # K 为所有样本的所有观测值集合，为了固定B和Pi，转为一个字典

    def _alpha(self, O: np.ndarray) -> np.ndarray:
        # O 为单条样本的观测序,已知O0...Ot，求qt和O的联合概率
        # 初始化 alpha
        alpha = np.zeros((len(O), len(self.N)))
        alpha[0] = self.Pi.reshape(-1) * self.B[:, self.K[O[0]]].reshape((1, -1))
        # 递推
        for t in range(1, len(O)):
            alpha[t] = np.matmul(alpha[t - 1], self.A) * self.B[:, self.K[O[t]]]
        return alpha

    def forward(self, O: np.ndarray) -> np.float:
        alpha = self._alpha(O)
        # 对alpha[-1]的qt相加就是前向概率
        return sum(alpha[-1])

    def _beta(self, O: np.ndarray) -> np.ndarray:
        # O 为单条样本的观测序列,已知Ot,t+1....tn，求qt和O的联合概率
        # 初始化 beta
        beta = np.zeros((len(O), len(self.N)))
        beta[-1] += 1
        # 递推
        for t in range(len(O) - 2, -1, -1):
            beta[t] = np.matmul(self.A, (beta[t + 1] * self.B[:, self.K[O[t]]]))
        return beta

    def backward(self, O: np.ndarray) -> np.ndarray:
        beta = self._beta(O)
        return np.matmul((self.B[:, self.K[O[0]]] * beta[0]), self.Pi).flat[0]

    def compute_po(self, O: np.ndarray, t) -> np.float:
        # 给定模型，求在t时刻状态为ot，整个O的概率
        if t == 0:
            return self.forward(O)
        if t == len(O) - 1:
            return self.backward(O)
        alpha = self._alpha(O)
        beta = self._beta(O)
        ot1 = self.K[O[t + 1]]
        po = np.matmul(alpha[t][np.newaxis, :],
                       np.matmul(self.A, (self.B[:, ot1][:, np.newaxis] * beta[t + 1][:, np.newaxis]))
                       ).flat
        return po[0]

    def compute_pq(self, O: np.ndarray, t: int) -> np.ndarray:
        # 给定模型和观测序列，求在时刻t时每个状态的概率
        gamma = self.compute_gamma(O)
        return gamma[t]

    def compute_pqi(self, O: np.ndarray, t: int, q) -> np.ndarray:
        # 给定模型和观测序列，求在时刻t时处于状态qi的概率
        gamma = self.compute_gamma(O, t)
        i = self.N[q]
        return gamma[t][i]

    def compute_ksi(self, O: np.ndarray, t: int) -> np.ndarray:
        # 给定模型和观测O，求在时刻t时处于状态qi，并且时刻t+1时处于状态qj的概率(每对ij的矩阵）
        alpha = self._alpha(O)
        beta = self._beta(O)
        ot1 = self.K[O[t + 1]]
        tmp = alpha[t][:, np.newaxis] * self.A * self.B[:, ot1][np.newaxis, :] * beta[t + 1, :]
        ksi = tmp / np.sum(tmp)
        return ksi

    def compute_p_next_q(self, O: np.ndarray, t: int, qi, qj) -> np.ndarray:
        # 给定模型和观测O，求在时刻t时处于状态qi，并且时刻t+1时处于状态qj的概率
        i = self.N[qi]
        j = self.N[qj]
        ksi = self.compute_ksi(O, t)
        return ksi[i][j]

    def compute_gamma(self, O: np.ndarray) -> np.float:
        # 给定模型和观测序列，求状态q出现的期望值(每个时刻每个q的概率的矩阵）
        alpha = self._alpha(O)
        beta = self._beta(O)

        tmp = alpha * beta
        gamma = tmp / np.sum(tmp, axis=1)[:, np.newaxis]
        return gamma

    def compute_e_qi(self, O: np.ndarray, q) -> np.float:
        # 给定模型和观测序列，求状态q出现的期望值
        i = self.N[q]
        gamma = self.compute_gamma(O)
        return sum(gamma[:, i])

    def compute_e_transfer_qi(self, O: np.ndarray, q) -> np.float:
        # 给定模型和观测序列，求由状态qi转移的期望值
        i = self.N[q]
        gamma = self.compute_gamma(O)
        return sum(gamma[:-1, i])

    def compute_e_transfer_qij(self, O: np.ndarray, qi, qj) -> np.float:
        # 求在观测O下，由状态i转移到状态j的期望
        return sum([self.compute_p_next_q(O, t, qi, qj) for t in range(len(O) - 1)])

    def fit_supervised(self, O: np.ndarray, I: np.ndarray) -> None:
        # O 为m个样本的观测序列, I 为m个样本对应的状态序列
        # m 为样本数, t 为观测序列长度
        N = dict(zip(set(I.flat), range(len(set(I.flat)))))
        K = dict(zip(set(O.flat), range(len(set(O.flat)))))
        m, t = O.shape
        A = np.array([[0 for i in range(len(N))] for j in range(len(N))])
        B = np.array([[0 for i in range(len(K))] for j in range(len(N))])
        Pi = np.array([0 for i in range(len(N))])

        for i in range(m):
            for j in range(1, t):
                t0 = I[i][j - 1]
                t1 = I[i][j]
                A[N[t0]][N[t1]] += 1
        A = A / np.sum(A, axis=1).reshape((-1, 1))

        for i in range(m):
            for j in range(t):
                t_O = O[i][j]
                t_I = I[i][j]
                B[N[t_I]][K[t_O]] += 1
        B = B / np.sum(B, axis=1).reshape((-1, 1))

        for i in range(m):
            t0_I = I[i][0]
            Pi[N[t0_I]] += 1
        Pi = Pi / np.sum(Pi)

        self.A = A
        self.B = B
        self.Pi = Pi
        self.N = N
        self.K = K

    def fit_unsupervised(self, O: np.ndarray, hidden_states: int, max_iter: int) -> None:
        # 《统计学习方法》有误，没说明是D条序列
        self.K = dict(zip(set(O.flat), range(len(set(O.flat)))))
        # 初始化参数
        self.N = dict(zip(range(hidden_states), range(hidden_states)))
        m, T = O.shape
        self.A = np.array([[np.random.random() for i in range(len(self.N))] for j in range(len(self.N))])
        self.B = np.array([[np.random.random() for i in range(len(self.K))] for j in range(len(self.N))])
        self.Pi = np.array([np.random.random()] * len(self.N))

        for _iter in range(max_iter):
            # 先计算alpha、beta、gamma、ksi等矩阵
            alpha = [self._alpha(_O) for _O in O]
            beta = [self._beta(_O) for _O in O]
            tmp = [alpha[i] * beta[i] for i in range(m)]
            gamma = [tmp[i] / np.sum(tmp[i], axis=1)[:, np.newaxis] for i in range(m)]
            ksi_lst = []
            for i in range(m):
                _ksi_lst = []  # 每个时刻的ksi
                for t in range(T - 1):
                    ot1 = self.K[O[i][t + 1]]
                    tmp = alpha[i][t][:, np.newaxis] * self.A * self.B[:, ot1][np.newaxis, :] * beta[i][t + 1, :]
                    ksi = tmp / np.sum(tmp)
                    _ksi_lst.append(ksi)
                _ksi_lst = np.sum(_ksi_lst, axis=0)
                ksi_lst.append(_ksi_lst)

            # 首先更新A
            for i in range(len(self.N)):
                deno = np.sum([np.sum(gamma[d][:-1, i]) for d in range(m)])
                for j in range(len(self.N)):
                    nume = np.sum([ksi_lst[d][i][j] for d in range(m)])
                    self.A[i][j] = nume / deno

            # 再更新B
            for j in range(hidden_states):
                deno = np.sum([gamma[i][:, j] for i in range(m)])
                for k in range(len(self.K)):
                    # 指示函数
                    indicator = np.array([[self.K[O[i][t]] == k for t in range(T)] for i in range(m)])
                    nume = np.sum(np.array([gamma[i][:, j] for i in range(m)]) * indicator)
                    self.B[j][k] = nume / deno

            # 再更新Pi
            self.Pi = np.sum([gamma[i][0] for i in range(m)], axis=0) / m

    def predict(self, O: np.ndarray) -> np.ndarray:
        # 初始化
        T = len(O)
        N = len(self.N)
        sigma = np.zeros((T, N))
        sigma[0] = self.Pi * self.B[:, self.K[O[0]]]
        psai = np.zeros((T, N))
        I = np.array([0] * T)
        # 动态规划
        for t in range(1, T):
            for i in range(N):
                sigma[t][i] = max([sigma[t - 1][j] * self.A[j][i] for j in range(N)]) * self.B[i][self.K[O[t]]]
                psai[t][i] = np.argmax([sigma[t - 1][j] * self.A[j][i] for j in range(N)])
        # 回溯
        I[-1] = int(np.argmax(sigma[-1]))
        for t in range(T - 2, -1, -1):
            I[t] = int(psai[t + 1][I[t + 1]])

        return I
