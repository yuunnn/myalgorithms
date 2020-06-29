from algs.algorithms.tree import AVLTree
from algs.utils import draw_tree
import random

tmp = random.sample(range(100), 50)

print("AVLTree:")
tree = AVLTree()
for i in tmp:
    tree.add(i)
print("in:", "AVL树的中序遍历也是一个有序数列")
print(tree.travel("in"))
draw_tree(tree.root, "AVLTree(with original 50 random numbers)")
print('################')
for i in tmp[:25]:
    tree.delete_value(i)
print("delete {}".format(','.join(map(str, tmp[:25]))))
print(tree.travel("in"))
draw_tree(tree.root, "AVLTree(deleted 25 random numbers)")
