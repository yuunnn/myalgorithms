import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from algs.ml_algorithms.fm import FM
from algs.utils import compute_mse

x, y = load_boston(return_X_y=1)

# 分桶为稀疏矩阵
######################################################################
x_new = []
for col in range(x.shape[1]):
    tmp = np.zeros([len(x), 16])
    _max = max(x[:, col])
    _min = min(x[:, col])
    for row in range(x.shape[0]):
        _ = int(round((x[row][col] - _min) / ((_max - _min) / 15), 0))
        tmp[row, _] = 1
    x_new.append(tmp)
x = np.concatenate(x_new, axis=1)
######################################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

fm = FM(k=15, learning_rate=0.03)
fm.fit(x_train, y_train)

y_predict = fm.predict(x_test.astype(np.float32))
print("验证集误差：", compute_mse(y_predict, y_test))
