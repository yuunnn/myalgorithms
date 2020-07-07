import numpy as np


def qr_factorization(a: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Gram-Schmidt 算法
    :param a: 被分解的矩阵
    :return: q正交基m*m,r上三角矩阵m*n

    首先Q[:,0]是A[:,0]的基，r[0,0]存放A[:,0]的模长
    然后用gs算法，每一列都递归的减去和上一列线性相关的部分，留下正交基，
    其实就是用A[:,j]减去和Q[:,j-1]里的每个列向量线性相关的部分，
    "减去线性相关的向量"就是b - np.dot(b,a) * (a / ||a||2),得到的就是和a正交的向量
    然后在r对角线上方存放内循环时上面一句话指的向量的模长
    在r[j][j]上存放A[:,j]和Q[:,:j-1]的所有列向量都正交的向量的模长
    """
    m, n = a.shape
    assert m >= n
    q = np.zeros([m, m])
    r = np.zeros([m, n])
    for j in range(n):
        v = a[:, j]
        for i in range(j):
            # r[i][j]存放a[:,j]上q[:, i]的投影长度
            r[i][j] = np.dot(q[:, i].T, a[:, j])
            # b - np.dot(b,a) * (a / ||a||2),得到的就是和a正交的向量
            v = v - r[i][j] * q[:, i]
        # r[j][j]存放最后一个和前面都正交的向量的模长
        r[j][j] = np.sum(v ** 2) ** 0.5
        q[:, j] = v / r[j][j]
    return q, r



