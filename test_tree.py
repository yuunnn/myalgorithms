from algs.algorithms.tree import BinaryTree
from algs.utils import draw_tree
import random

tmp = random.sample(range(100), 50)

print("BinaryTree:")
tree = BinaryTree()
for i in tmp:
    tree.add(i)
print("pre:")
print(tree.travel("pre"))
print('################')
print("in:")
print(tree.travel("in"))
print('################')
print("post:")
print(tree.travel("post"))
print('################')
print("leaves:")
print(tree.leaves)
draw_tree(tree.root, "BinaryTree")