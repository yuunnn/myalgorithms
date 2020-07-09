from typing import List, Union, Iterator, Generator


def quick_sort_fp(lst: List[(Union[int, float])]) -> List[(Union[int, float])]:
    if lst.__len__() <= 1:
        return lst
    left, right = [], []
    for i in lst[1:]:
        left.append(i) if i <= lst[0] else right.append(i)
    return quick_sort_fp(left) + [lst[0]] + quick_sort_fp(right)


def quick_sort_normal(lst: List[(Union[int, float])]) -> None:
    def _sort(_lst, low, high):
        if high - low == 1:
            if _lst[low] > _lst[high]:
                _lst[low], _lst[high] = _lst[high], _lst[low]
                return
        if high - low <= 0:
            return
        left = low
        right = high
        tmp = _lst[low]
        while left < right:
            while left < right:
                if _lst[right] < tmp:
                    _lst[left] = _lst[right]
                    break
                right -= 1
            while left < right:
                if _lst[left] > tmp:
                    _lst[right] = _lst[left]
                    break
                left += 1
        _lst[left] = tmp

        _sort(_lst, low, left - 1)
        _sort(_lst, left + 1, high)

    _sort(lst, 0, len(lst) - 1)
