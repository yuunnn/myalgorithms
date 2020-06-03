from algs.ml_algorithms.hmm import Hmm
import numpy as np

# 前向概率
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res = hmm.forward(np.array(O))
print("前向概率", res)
print("############################################")

# 后项概率
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res = hmm.backward(np.array(O))
print("后向概率", res)
print("############################################")

# 有监督学习
O = [[0, 1, 0, 1, 0], [1, 1, 0, 1, 0]]
I = [[0, 1, 2, 3, 2], [1, 0, 2, 2, 3]]
hmm = Hmm()
hmm.fit_supervised(np.array(O), np.array(I))
print("有监督学习")
print("状态转移矩阵", hmm.A)
print("观测概率矩阵", hmm.B)
print("初始状态概率向量", hmm.Pi)
print("############################################")

