from algs.nn_algoritms.auto_diff import Var, gradient

print("一个'玩具型'自动微分工具，本质上是python的重载操作符加上递归实现了forward和链式求导")
a = Var(1.5)
b = Var(2.5)
c = Var(3)
d = Var(4)
e = Var(5)

o1 = a * b
o2 = o1 + c
o3 = o2 / d
o4 = o2 - e
o5 = o4 * o3

res = gradient(a, o5)

print("a = 1.5, b = 2.5, c = 3, d = 4, e = 5")
print("o1 = a * b")
print("o2 = o1 + c")
print("o3 = o2 / d")
print("o4 = o2 - e")
print("o5 = o4 * o3")
print("求o5对a的偏导：")
print(res)
