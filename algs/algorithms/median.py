def median(lst):
    workers = 4
    _reduce_number = 100000

    def _map(number=0, counts=0, _lst=[], is_right=True):
        if len(_lst) == 1:
            return _lst[0]
        if len(_lst) == 2:
            return (_lst[0] + _lst[1]) / 2

        count_lefts, count_rights = 0, 0
        lefts, rights = [], []
        for i in range(workers):
            count_left, count_right, left, right = _reduce(lst[i * _reduce_number:(i + 1) * _reduce_number])
            count_lefts += count_left
            count_rights += count_right
            lefts += left
            rights += right

        number = count_lefts + count_rights
        if count_lefts < count_rights:
            _m = count_rights[0]
            _number = number - count_lefts
            _map(number=_m, counts=_number, _lst=rights)
        else:
            _m = count_rights[0]
            _number = number - count_rights
            _map(number=_m, counts=_number, _lst=lefts)

    def _reduce(l, m):
        left = []
        right = []
        count_left = 0
        count_right = 0
        for i in l[1:]:
            if i < m:
                left.append(i)
                count_left += 1
            else:
                right.append(i)
                count_right += 1
        return count_left, count_right, left, right
