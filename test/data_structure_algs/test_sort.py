import random
from algs.data_structure_algorithms.sort import quick_sort_fp, quick_sort_normal
import time

lst = random.sample(range(2000000), 1000000)
print("比较两种快排实现的时间，对随机的100万个整数进行排序")

a = time.time()
sorted_lst = quick_sort_fp(lst)
b = time.time()
quick_sort_normal(lst)
c = time.time()

print("第一种快排是haskel时'偷'来的，非常好玩，优点是实现方式非常人性化……缺点是需要创建新的list，有空间上的开销")
print("第一种的时间")
print(b - a)
print("第二种快排是《算法》经典快排,原地排序")
print("第二种的时间")
print(c - b)
print("可以看到，在100万的数量级下，两者差不多，甚至第一种可能还会快一点点")
print("验证准确")
print(lst == sorted_lst)
