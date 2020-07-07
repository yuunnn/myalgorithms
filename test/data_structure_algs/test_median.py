import random
import numpy as np
import time
from algs.data_structure_algorithms.median import median


print("用多进程模拟分布式计算1000万个数的中位数,worker数量=2")
lst = random.sample(range(50000000), 10000000)
a = time.time()
res = median(lst, workers_number=2, map_volume=5000000)
b = time.time()
res2 = np.median(lst)
c = time.time()

print("###################################")
print("结果:", '\n', res, '\n', "时间:", '\n', b - a)
print("###################################")
print("验证准确性，numpy结果:", "\n", res2, '\n', "时间:", '\n', c - b)
