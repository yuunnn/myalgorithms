from algs.data_structure_algorithms.tree import RBTree
from algs.utils import draw_tree
import random

tmp = random.sample(range(500), 100)

print("RBTree:")
tree = RBTree()
for i in tmp:
    tree.add(i)
print("in:", "RB树的中序遍历也是一个有序数列")
print(tree.travel("in"))
draw_tree(tree.root, "RBTree(with original 100 random numbers)")
for i in tmp[:50]:
    tree.delete_value(i)
draw_tree(tree.root, "RBTree(deleted 50 random numbers)")
