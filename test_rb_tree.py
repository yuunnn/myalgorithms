from algs.algorithms.tree import RBTree
from algs.utils import draw_tree
import random

# tmp = random.sample(range(500), 100)
tmp = [5,2,1,3,4,7,6,8,9,10]
print("RBTree:")
tree = RBTree()
for i in tmp:
    print(i)
    tree.add(i)
print("in:", "RB树的中序遍历也是一个有序数列")
print(tree.travel("in"))
# tree.delete_min()
# tree.delete_value(6)
# draw_tree(tree.root, "RBTree(with original 100 random numbers)")
tree.delete_value(6)
# draw_tree(tree.root, "RBTree(with original 100 random numbers)")

