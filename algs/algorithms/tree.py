class Node:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:

    def __init__(self):
        self.root = None
        self.nodelist = []

    def add(self, value):
        node = Node(value)
        if self.root is None:
            self.root = node
            self.nodelist.append(self.root)
        else:
            point = self.nodelist[0]
            if point.left is None:
                point.left = node
                self.nodelist.append(point.left)
                return
            elif point.right is None:
                point.right = node
                self.nodelist.append(point.right)
                self.nodelist.pop(0)
                return

    @staticmethod
    def pre_order_travel(node, _res):
        if node is None:
            return None
        _res.append(node.value)
        BinaryTree.pre_order_travel(node.left, _res)
        BinaryTree.pre_order_travel(node.right, _res)

    @staticmethod
    def in_order_travel(node, _res):
        if node is None:
            return None
        BinaryTree.in_order_travel(node.left, _res)
        _res.append(node.value)
        BinaryTree.in_order_travel(node.right, _res)

    @staticmethod
    def post_order_travel(node, _res):
        if node is None:
            return None
        BinaryTree.post_order_travel(node.left, _res)
        BinaryTree.post_order_travel(node.right, _res)
        _res.append(node.value)

    def travel(self, method="pre"):
        _res = []
        if method == "pre":
            self.pre_order_travel(self.root, _res)
        if method == "in":
            self.in_order_travel(self.root, _res)
        if method == "post":
            self.post_order_travel(self.root, _res)
        return _res

    @property
    def leaves(self):
        _leaves = []

        def _travel(node, _leaves):
            if node is None:
                return None
            if node.left is None and node.right is None:
                _leaves.append(node.value)
            if node.left is not None:
                _travel(node.left, _leaves)
            if node.right is not None:
                _travel(node.right, _leaves)

        _travel(self.root, _leaves)

        return _leaves

    @property
    def height(self):

        def _travel(node):
            if node.value is None:
                return 0
            if node.left is None and node.right is None:
                return 1
            if node.left is None and node.right is not None:
                return 1 + _travel(node.right)
            if node.left is not None and node.right is None:
                return 1 + _travel(node.left)
            if node.left is not None and node.right is not None:
                return 1 + max(_travel(node.left), _travel(node.right))

        return _travel(self.root)

