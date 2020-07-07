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
print(type(res))
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

# 联合前后项概率
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res0 = hmm.compute_po(np.array(O), t=0)
print("联合前后项概率,t=0")
print(res0)
res1 = hmm.compute_po(np.array(O), t=1)
print("联合前后项概率,t=1")
print(res1)
res2 = hmm.compute_po(np.array(O), t=2)
print("联合前后项概率,t=2")
print(res2)
print("############################################")

# 给定模型和观测序列，求在时刻t时每个状态q的概率
print("给定模型和观测序列，求在时刻t时处于状态qi的概率")
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res0 = hmm.compute_pq(O, t=0)
print(res0)
print("############################################")

# 给定模型和观测序列，求状态q出现的期望值
print("给定模型和观测序列，求状态q出现的期望值")
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res = hmm.compute_e_qi(O, 1)
print(res)
print("############################################")

# 给定模型和观测序列，求由状态qi转移的期望值
print("给定模型和观测序列，求由状态qi转移的期望值")
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res = hmm.compute_e_transfer_qi(O, 1)
print(res)

# 给定模型和观测O，求在时刻t时处于状态qi，并且时刻t+1时处于状态qj的概率
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res = hmm.compute_p_next_q(O, 0, 0, 1)
# print(res)
print("############################################")

# 求在观测O下，由状态i转移到状态j的期望
print("求在观测O下，由状态i转移到状态j的期望")
Q = {0, 1, 2}
O = [0, 1, 0]
N = dict(zip(set(Q), range(len(set(Q)))))
K = dict(zip(set(O), range(len(set(O)))))
hmm = Hmm(A=np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]),
          B=np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]),
          Pi=np.array([0.2, 0.4, 0.4]).reshape((-1, 1)),
          N=N, K=K)
res = hmm.compute_e_transfer_qij(O, 1, 1)
print(res)
print("############################################")

# 有监督学习
O = [[0, 1, 2, 1, 0, 1], [1, 2, 0, 1, 0, 0], [0, 2, 1, 1, 0, 2], [1, 1, 1, 0, 2, 0], [1, 0, 2, 1, 1, 1],
     [2, 0, 1, 0, 1, 0]]
I = [[0, 1, 2, 1, 2, 0], [1, 0, 2, 2, 0, 1], [1, 2, 1, 2, 2, 0], [2, 0, 0, 1, 2, 1],
     [0, 1, 0, 2, 1, 2], [2, 2, 0, 1, 1, 0]]
hmm = Hmm()
hmm.fit_supervised(np.array(O), np.array(I))
print("有监督学习")
print("状态转移矩阵\n", hmm.A)
print("观测概率矩阵\n", hmm.B)
print("初始状态概率向量", hmm.Pi)
print("############################################")

# 无监督学习
O = [[0, 1, 2, 1, 0, 1], [1, 2, 0, 1, 0, 0], [0, 2, 1, 1, 0, 2], [1, 1, 1, 0, 2, 0], [1, 0, 2, 1, 1, 1],
     [2, 0, 1, 0, 1, 0]]
hmm = Hmm()
hmm.fit_unsupervised(np.array(O), 3, 50)
print("无监督学习,Baum-Welch算法")
print("状态转移矩阵\n", np.round(hmm.A, 3))
print("观测概率矩阵\n", np.round(hmm.B, 3))
print("初始状态概率向量", np.round(hmm.Pi, 3))
print("############################################")


# 预测，维特比算法
O = [[0, 1, 2, 1, 0, 1], [1, 2, 0, 1, 0, 0], [0, 2, 1, 1, 0, 2], [1, 1, 1, 0, 2, 0], [1, 0, 2, 1, 1, 1],
     [2, 0, 1, 0, 1, 0]]
I = [[0, 1, 2, 1, 2, 0], [1, 0, 2, 2, 0, 1], [1, 2, 1, 2, 2, 0], [2, 0, 0, 1, 2, 1],
     [0, 1, 0, 2, 1, 2], [2, 2, 0, 1, 1, 0]]
hmm = Hmm()
hmm.fit_supervised(np.array(O), np.array(I))
res = hmm.predict(np.array([1, 1, 1, 0, 2, 0]))
print("预测，维特比算法")
print("观测序列为:\n", np.array([1, 1, 1, 0, 2, 0]))
print("预测状态序列:")
print(res)
