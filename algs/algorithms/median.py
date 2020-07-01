from concurrent.futures import ThreadPoolExecutor
import random


def _reduce(l, m):
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


def _first_reduce(l, m):
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
    workers = 4
    _reduce_number = 100000
    executor = ThreadPoolExecutor(workers)

    def _map(_lst=[], is_first=False, target=0, total=0):
        if len(_lst) == 1:
            executor.shutdown()
            return _lst[0]
        if len(_lst) == 2:
            executor.shutdown()
            print("fffff")
            print(_lst)
            if target.is_integer():
                return _lst[0]
            return (_lst[0] + _lst[1]) / 2

        lefts, rights = [], []
        try:
            _m = random.sample(_lst, 1)[0]
        except:
            print(_lst)
        if is_first:
            total_left = 0
            total_right = 0
            futures = []
            for i in range(workers):
                if i == workers - 1:
                    futures.append(executor.submit(_first_reduce, l=_lst[i * _reduce_number:], m=_m))
                else:
                    futures.append(
                        executor.submit(_first_reduce, l=_lst[i * _reduce_number:(i + 1) * _reduce_number], m=_m))
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
                    futures.append(executor.submit(_reduce, l=_lst[i * _reduce_number:], m=_m))
                futures.append(executor.submit(_reduce, l=_lst[i * _reduce_number:(i + 1) * _reduce_number], m=_m))
            for future in futures:
                count_left, left, right = future.result()
                lefts += left
                rights += right
                total_left += count_left

        if total_left <= target:
            total = total - total_left
            target = target - total_left
            return _map(_lst=rights, total=total, target=target)
        else:
            total = total_left
            return _map(_lst=lefts, total=total, target=target)

    return _map(_lst=lst, is_first=True)
