import numpy as np
import pandas as pd


def compute_mse(y1, y2):
    if len(y1) != len(y2):
        raise ValueError("y1.length != y2.length")
    y1 = list(y1)
    y2 = list(y2)
    return sum([(y1[i] - y2[i]) ** 2 for i in range(len(y1))]) / len(y1)


def compute_mae(y1, y2):
    if len(y1) != len(y2):
        raise ValueError("y1.length != y2.length")
    y1 = list(y1)
    y2 = list(y2)
    return sum([abs(y1[i] - y2[i]) for i in range(len(y1))]) / len(y1)


def compute_logloss(ypred, ytrue):
    if len(ypred) != len(ytrue):
        raise ValueError("y1.length != y2.length")
    return -np.mean([np.log(ypred[i][ytrue[i]]) for i in range(len(ypred))])


def compute_confusion_matrix(ypred, yture):
    if len(ypred) != len(yture):
        raise ValueError("y1.length != y2.length")
    return pd.crosstab(ypred, yture)


def standardization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw_tree(node):
    # 该代码来源于https://zhuanlan.zhihu.com/p/35574577
    import networkx as nx
    import matplotlib.pyplot as plt

    def create_graph(G, node, pos={}, x=0, y=0, layer=1):
        pos[node.value] = (x, y)
        if node.left:
            G.add_edge(node.value, node.left.value)
            l_x, l_y = x - 1 / 2 ** layer, y - 1
            l_layer = layer + 1
            create_graph(G, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
        if node.right:
            G.add_edge(node.value, node.right.value)
            r_x, r_y = x + 1 / 2 ** layer, y - 1
            r_layer = layer + 1
            create_graph(G, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
        return G, pos

    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(8, 10))  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=300)
    plt.show()
