class Node:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class Btree:

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
        Btree.pre_order_travel(node.left, _res)
        Btree.pre_order_travel(node.right, _res)

    @staticmethod
    def in_order_travel(node, _res):
        if node is None:
            return None
        Btree.in_order_travel(node.left, _res)
        _res.append(node.value)
        Btree.in_order_travel(node.right, _res)

    @staticmethod
    def post_order_travel(node, _res):
        if node is None:
            return None
        Btree.post_order_travel(node.left, _res)
        Btree.post_order_travel(node.right, _res)
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


t = Btree()
L = [1, 2, 3, 4, 5]
for i in L:
    t.add(i)
res = t.travel(method="pre")
print(res)
