from algs.algorithms.tree import RBTree
from algs.utils import draw_tree
import random

# tmp = random.sample(range(100), 20)
tmp = [5,2,1,3,4,6,7,8,9,10]
print("RBTree:")
tree = RBTree()
for i in tmp:
    print(i)
    tree.add(i)
print("in:", "RB树的中序遍历也是一个有序数列")
print(tree.travel("in"))
print(tree.root.color)
draw_tree(tree.root, "RBTree(with original 100 random numbers)")
# print('################')
# for i in tmp[:50]:
#     tree.delete_value(i)
# print("delete {}".format(','.join(map(str, tmp[:50]))))
# print(tree.travel("in"))
# draw_tree(tree.root, "AVLTree(deleted 50 random numbers)")