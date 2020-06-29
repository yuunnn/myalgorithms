from algs.algorithms.tree import BinarySortTree
from algs.utils import draw_tree
import random

tmp = random.sample(range(100), 50)

print("BinarySortTree:")
tree = BinarySortTree()
for i in tmp:
    tree.add(i)
print("in:", "二叉查找树的中序遍历是一个有序数列")
print(tree.travel("in"))
print('################')
print("minimum:")
print(tree.minimum)
print('################')
print("maximum:")
print(tree.maximum)
print('################')
print("{}'s successor value".format(tmp[25]))
print(tree.get_successor_value(tmp[25]))
print('################')
print("{}'s predecessor value".format(tmp[25]))
print(tree.get_predecessor_value(tmp[25]))
draw_tree(tree.root, "BinarySortTree(with original 50 random numbers)")
print('################')
for i in tmp[:25]:
    tree.delete_value(i)
print("delete {}".format(','.join(map(str,tmp[:25]))))
print(tree.travel("in"))
draw_tree(tree.root, "BinarySortTree(deleted 25 random numbers)")

