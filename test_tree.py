from algs.algorithms.tree import BinaryTree, Node

tree = BinaryTree()
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    tree.add(i)
print(tree.travel("pre"))
print('########')
print(tree.travel("in"))
print('########')
print(tree.travel("post"))
print('########')
print(tree.leaves)
