"""
简单的一个dag顺序遍历算法，还有优化空间，用法见test

"""


def dag(data):
    res = {}
    node = []
    values = []

    for v in data.values():
        values.extend(v)

    for k in data.keys():
        if k not in values:
            node.append(k)

    def recursion(_node, _data):
        _res = {}
        if not _data[_node]:
            return _res
        for i in _data[_node]:
            _res[i] = recursion(i, _data)
        return _res

    for i in node:
        res[i] = recursion(i, data)
    return res


def sort_dag(data):
    res = []
    values = []
    for v in data.values():
        values.extend(v)

    def recursion(data, res, values):
        if len(data) == 0:
            return res

        for k in list(data.keys()):
            if k not in values:
                res.append(k)
                # 更新下values
                for v in data[k]:
                    values.remove(v)
                # 删除没有出度的节点
                del data[k]

        return recursion(data, res, values)

    return list(reversed(recursion(data, res, values)))