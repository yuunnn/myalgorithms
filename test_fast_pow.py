import time
from algs.algorithms.fast_pow import fast_pow, matrix_fast_pow


A = time.time()
res = fast_pow(520, 1314)
B = time.time()
print('快速幂算520的1314次方')
print(res)
print('时间')
print(B-A)
print('######################################################################')


def vector_matmul(A, B):
    return [A[0] * B[0][0] + A[1] * B[0][1], A[0] * B[1][0] + A[1] * B[1][1]]


def fibo(n):
    B = [[0, 1], [1, 1]]
    A = [0, 1]
    B = matrix_fast_pow(B, n - 3)
    return vector_matmul(A, B)[0]


A = time.time()
for i in range(100):
    res = fibo(50000)
B = time.time()
print('利用矩阵快速幂算100次斐波那契数列第50000个数')
print(res)
print('时间')
print(B - A)
