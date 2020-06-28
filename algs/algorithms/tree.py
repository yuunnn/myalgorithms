from typing import List, Union


class Node:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.p = None


class BinaryTree:

    def __init__(self):
        self.root = None
        self.nodelist = []

    def add(self, value: Union[int, float]) -> None:
        node = Node(value)
        if self.root is None:
            self.root = node
            self.nodelist.append(self.root)
        else:
            point = self.nodelist[0]
            if point.left is None:
                point.left = node
                point.left.p = point
                self.nodelist.append(point.left)
                return
            elif point.right is None:
                point.right = node
                point.right.p = point
                self.nodelist.append(point.right)
                self.nodelist.pop(0)
                return

    @staticmethod
    def pre_order_travel(node: Node, _res: List) -> None:
        if node is None:
            return None
        _res.append(node.value)
        BinaryTree.pre_order_travel(node.left, _res)
        BinaryTree.pre_order_travel(node.right, _res)

    @staticmethod
    def in_order_travel(node: Node, _res: List) -> None:
        if node is None:
            return None
        BinaryTree.in_order_travel(node.left, _res)
        _res.append(node.value)
        BinaryTree.in_order_travel(node.right, _res)

    @staticmethod
    def post_order_travel(node: Node, _res: List) -> None:
        if node is None:
            return None
        BinaryTree.post_order_travel(node.left, _res)
        BinaryTree.post_order_travel(node.right, _res)
        _res.append(node.value)

    def travel(self, method: str = "pre") -> List[Union[int, float]]:
        _res = []
        if method == "pre":
            self.pre_order_travel(self.root, _res)
        if method == "in":
            self.in_order_travel(self.root, _res)
        if method == "post":
            self.post_order_travel(self.root, _res)
        return _res

    @property
    def leaves(self) -> List[Union[int, float]]:
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
    def height(self) -> int:

        def _travel(node: Node) -> None:
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


class BinarySortTree(BinaryTree):
    def __init__(self):
        super().__init__()

    def add(self, value: Union[int, float]):
        node = Node(value)
        if self.root is None:
            self.root = node
            return
        point = self.root
        tmp_node = None
        while point is not None:
            tmp_node = point
            if value < point.value:
                point = point.left
            else:
                point = point.right
        if value < tmp_node.value:
            tmp_node.left = node
        else:
            tmp_node.right = node
        node.p = tmp_node

    @staticmethod
    def _search(node: Node, value: Union[int, float]) -> Node:
        while node is not None and value != node.value:
            if value < node.value:
                node = node.left
            else:
                node = node.right
        return node

    def search(self, value: Union[int, float]) -> Node:
        return self._search(self.root, value)

    @staticmethod
    def get_minimum_node(node: Node) -> Node:
        while node.left is not None:
            node = node.left
        return node

    @property
    def minimum(self) -> Union[int, float]:
        node = self.root
        return self.get_minimum_node(node).value

    @staticmethod
    def get_maximum_node(node: Node) -> Node:
        while node.right is not None:
            node = node.right
        return node

    @property
    def maximum(self) -> Union[int, float]:
        node = self.root
        return self.get_maximum_node(node).value

    def get_successor_node(self, node: Node) -> Node:
        if node.right is not None:
            return self.get_minimum_node(node.right)
        else:
            point = node.p
            while point is not None and node.value == point.right.value:
                node = point
                point = point.p
            if point.value == self.root.value:
                raise ValueError("this node has not a successor node")
            return point

    def get_successor_value(self, value: Union[int, float]) -> Union[int, float]:
        node = self.search(value)
        return self.get_successor_node(node).value

    def get_predecessor_node(self, node: Node) -> Node:
        if node.left is not None:
            return self.get_maximum_node(node.left)
        else:
            point = node.p
            while point.p is not None and point.left.value == node.value:
                node = point
                point = node.p
            if point.value == self.root.value:
                raise ValueError("this node has not a predecessor node")
            return point

    def get_predecessor_value(self, value: Union[int, float]) -> Union[int, float]:
        node = self.search(value)
        return self.get_predecessor_node(node).value

    def transplant(self, node_u: Node, node_v: Node) -> Node:
        if node_u.p is None:
            self.root = node_v
        elif node_u.p.left is not None and node_u.value == node_u.p.left.value:
            node_u.p.left = node_v
        else:
            node_u.p.right = node_v
        if node_v is not None:
            node_v.p = node_u.p

    def delete_node(self, node: Node) -> Node:
        if node.left is None:
            self.transplant(node, node.right)
        elif node.right is None:
            self.transplant(node, node.left)
        else:
            tmp_node = self.get_minimum_node(node.right)
            if tmp_node.p.value != node.value:
                self.transplant(tmp_node, tmp_node.right)
                tmp_node.right = node.right
                tmp_node.right.p = tmp_node
            self.transplant(node, tmp_node)
            tmp_node.left = node.left
            tmp_node.left.p = tmp_node

    def delete_value(self, value: Union[int, float]) -> Node:
        node = self.search(value)
        self.delete_node(node)
