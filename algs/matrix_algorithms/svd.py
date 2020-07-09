import numpy as np
from .qr_factorization import qr_factorization

"""
首先对矩阵A进行svd分解的形式为：
A = UVD
AAT = U V D Dt Vt Ut = U V Vt Ut
ATA = Dt Vt Ut U V D Dt = Dt Vt V Dt
可以看出来和AAT和ATA的特征值分解形式类似，因此，
一个矩阵A的svd分解，u为AAT的特征向量，d为ATA的特征向量，v为任意一个的特征根的开方

然后AAT和ATA的特征值分解可以用QR分解来做：

首先，矩阵A的特征向量X和特征值lambda满足：
AX = lambda X
而
A 可以被分解为QR，所以
A = QR
Q.T * A = Q.T * Q * R = R
Q.T * A * Q = RQ
Q.T * A * Q 不改变A的特征值,所以RQ不改变A的特征值
令A1 = R0Q0，A1的特征值=A的特征值
直到An = R(n-1)Q(n-1)，此时An为一个对角矩阵（为什么能收敛？），
对角矩阵的特征值就是对角元素（why？）
得到A的特征值后，代入AX = lambda X，解方程组，得到特征向量（或者用近似解法）
"""


def is_diag(a: np.ndarray) -> bool:
    M, N = a.shape
    for m in range(M):
        for n in range(N):
            if m != n and a[m][n] > 0.00001:
                return False
    return True


def solve(a: np.ndarray) -> np.ndarray:
    # 解法来源：https://zhuanlan.zhihu.com/p/97033779
    # 证明：（看不懂）
    # http://homepage.divms.uiowa.edu/~atkinson/m171.dir/sec_9-7.pdf
    x = np.random.random(len(a))
    inv_a = np.linalg.inv(a)
    for i in range(3):
        x = np.matmul(inv_a, x)
    return x / np.sum(x ** 2) ** 0.5


def solve_eig_f(a: np.ndarray, eig_v: np.ndarray) -> np.ndarray:
    assert a.shape[0] == a.shape[1]
    res = np.zeros(a.shape)
    e = np.eye(a.shape[0])
    for i in range(len(res)):
        if i >= len(eig_v):
            break
        vec = solve(a - eig_v[i] * e)
        res[i] = vec
    return res


def eig(a: np.ndarray) -> np.ndarray:
    assert a.shape[0] == a.shape[1]
    q, r = qr_factorization(a)
    a_new = np.matmul(r, q)
    while not is_diag(a_new):
        q, r = qr_factorization(a_new)
        a_new = np.matmul(r, q)
    res = []
    for m in range(len(a)):
        res.append(a_new[m][m])
    res = sorted(res, reverse=True)
    return np.array(res)


def svd(a: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    aat = np.matmul(a, a.T)
    ata = np.matmul(a.T, a)

    if aat.shape[0] > ata.shape[0]:
        eig_value = eig(ata)
    else:
        eig_value = eig(aat)

    f_aat = solve_eig_f(aat, eig_value)
    f_ata = solve_eig_f(ata, eig_value)

    eig_value = np.diag(eig_value ** 0.5)
    if len(f_aat) > len(eig_value):
        eig_value = np.concatenate([eig_value, np.zeros([len(f_aat) - len(f_ata), len(f_ata)])], axis=0)
    elif len(f_ata) > len(eig_value):
        eig_value = np.concatenate([eig_value, np.zeros([len(f_aat), len(f_ata) - len(f_aat)])], axis=1)
    return f_aat.T, eig_value, f_ata
