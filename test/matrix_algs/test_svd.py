import numpy as np
from algs.matrix_algorithms.svd import svd

a = np.random.random([5, 3])
u, v, d = svd(a)
print("原矩阵:")
print(a)
print("奇异值")
print(v)
print("U*V*D")
print(np.matmul(np.matmul(u, v), d))

