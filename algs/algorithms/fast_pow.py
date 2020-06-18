def multyply(A, B):
    res = [[0 for i in range(len(A))] for j in range(len(B))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            res[i][j] = sum([A[i][k] * B[k][j] for k in range(len(B))])
    return res


def fast_pow(a, m):
    c = 1
    while m > 0:
        if m & 1:
            c *= a
        a = a * a
        m >>= 1
    return c


def matrix_fast_pow(a, m):
    c = [[1 for _i in range(len(a))] for _j in range(len(a[0]))]
    while m > 0:
        if m & 1:
            c = multyply(c, a)
        a = multyply(a, a)
        m >>= 1
    return c
