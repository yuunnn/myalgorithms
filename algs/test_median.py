from algs.algorithms.median import median
import random
import numpy as np
import time

lst = random.sample(range(400000), 400000)
# lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
a = time.time()
res = median(lst)
b = time.time()
res2 = np.median(lst)
c = time.time()

print("用多进程模拟分布式计算中位数")
print("结果:", '\n', res, b-a)
print("验证准确性，numpy结果:", "\n", res2,c-b)
