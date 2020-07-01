from algs.algorithms.median import median
import random
import numpy as np
import time

lst = random.sample(range(20000000), 1000000)
a = time.time()
res = median(lst)

b = time.time()
res2 = np.median(lst)
c = time.time()

print("用多进程模拟分布式计算100万个数的中位数")
print("###################################")
print("结果:", '\n', res,'\n', "时间:", '\n', b - a)
print("###################################")
print("验证准确性，numpy结果:", "\n", res2,'\n', "时间:", '\n', c - b)
