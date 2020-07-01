from concurrent.futures import ProcessPoolExecutor
import random
"""
1种线性时间排序算法求中位数的算法，世间复杂度为O(N), 约数2O(N),可以转化用分布式方法求
算法：
1 在第1次reduce后统计total，target = (total+1) / 2 + 0.5
2 每次reduce后都计算left个数，如果left个数 < target：中位数在right中，否则在left中
2 target更新规则：left区间target不变，right区间为 target - left
（target - left）if (target - left) > 0 else 新区间大小 + (target - left)
4 边界条件有些复杂，大概有5种情况
"""


def _map(l, m):
    left = []
    right = []
    count_left = 0
    for i in l:
        if i <= m:
            left.append(i)
            count_left += 1
        else:
            right.append(i)
    return count_left, left, right


def _first_map(l, m):
    left = []
    right = []
    count_left = 0
    count_right = 0
    for i in l:
        if i <= m:
            left.append(i)
            count_left += 1
        else:
            right.append(i)
            count_right += 1
    return count_left, count_right, left, right


def median(lst):
    workers = 2
    _map_number = 500000
    executor = ProcessPoolExecutor(workers)

    def _reduce(_lst=[], is_first=False, target=0.0):
        if len(_lst) == 1:
            executor.shutdown()
            return _lst[0]
        if len(_lst) == 2:
            executor.shutdown()
            if target.is_integer():
                return min(_lst)
            return (_lst[0] + _lst[1]) / 2

        lefts, rights = [], []
        _m = random.sample(_lst, 1)[0]
        if is_first:
            total_left = 0
            total_right = 0
            futures = []
            for i in range(workers):
                if i == workers - 1:
                    futures.append(executor.submit(_first_map, l=_lst[i * _map_number:], m=_m))
                else:
                    futures.append(
                        executor.submit(_first_map, l=_lst[i * _map_number:(i + 1) * _map_number], m=_m))
            for future in futures:
                count_left, count_right, left, right = future.result()
                total_left += count_left
                total_right += count_right
                lefts += left
                rights += right
            total = total_left + total_right
            target = total / 2 + 0.5
        else:
            total_left = 0
            futures = []
            for i in range(workers):
                if i == workers - 1:
                    futures.append(executor.submit(_map, l=_lst[i * _map_number:], m=_m))
                else:
                    futures.append(executor.submit(_map, l=_lst[i * _map_number:(i + 1) * _map_number], m=_m))
            for future in futures:
                count_left, left, right = future.result()
                lefts += left
                rights += right
                total_left += count_left

        if total_left == target and target.is_integer():
            return _m
        elif total_left < target:
            target = target - total_left
            if target == 0.5:
                return (max(lefts) + min(rights)) / 2
            return _reduce(_lst=rights, target=target)
        else:
            if target == 0.5:
                return (max(lefts) + min(rights)) / 2
            return _reduce(_lst=lefts, target=target)

    return _reduce(_lst=lst, is_first=True)
