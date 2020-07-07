import numpy as np
from algs.matrix_algorithms.qr_factorization import qr_factorization

a = np.random.random([4, 4])
q, r = qr_factorization(a)

print("原矩阵")
print(a)
print("q")
print(q)
print("r")
print(r)
print("q*r")
print(np.matmul(q, r))
print("###############验证##################")
print("###############验证##################")
q, r = np.linalg.qr(a)
print("numpy算法q")
print(q)
print("numpy算法r")
print(r)
print("numpy算法q*r")
print(np.matmul(q, r))
