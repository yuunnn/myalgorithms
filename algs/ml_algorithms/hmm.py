import numpy as np


class Hmm:
    def __init__(self, A=None, B=None, Pi=None, N=None, K=None):
        self.A = A  # 状态转移矩阵
        self.B = B  # 观测概率矩阵
        self.Pi = Pi  # 初始状态概率向量
        self.N = N  # N 为所有样本的所有状态集合,为了便于固定A和Pi，转为一个字典
        self.K = K  # K 为所有样本的所有观测值集合，为了固定B和Pi，转为一个字典

    def forward(self, O: np.ndarray) -> np.float:
        # O 为单条样本的观测序列
        # 初始化 alpha,t时刻，已知0...t的观测序列的，求qt和O的联合概率
        alpha = np.zeros((len(O), len(self.N)))
        alpha[0] = self.Pi.reshape(-1) * self.B[:, self.K[O[0]]].reshape((1, -1))
        # 递推
        for t in range(1, len(O)):
            alpha[t] = np.matmul(alpha[t - 1], self.A) * self.B[:, self.K[O[t]]]
        # 对所有的qt相加就是前向概率
        return np.sum(alpha[-1])

    def backward(self, O: np.ndarray) -> np.float:
        # O 为单条样本的观测序列
        # 初始化 beta,t时刻，已知qt,求t,t+1....tn的后项概率
        beta = np.zeros((len(O), len(self.N)))
        beta[-1] += 1
        # 递推
        for t in range(len(O) - 2, -1, -1):
            beta[t] = np.matmul(beta[t + 1], self.A) * self.B[:, self.K[O[t]]]
        return np.sum(beta[0])

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
        Pi / np.sum(Pi)

        self.A = A
        self.B = B
        self.Pi = Pi
        self.N = N
        self.K = K

    def fit_unsupervised(self, O: np.ndarray) -> None:

        def _e_step():
            pass

        def _m_step():
            pass

        pass
