import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Process, Queue

"""
1种线性时间排序算法求中位数的算法，时间复杂度为O(N)级别,可以转化用分布式方法求
算法：
1 在第1次reduce后统计total，target = (total+1) / 2 + 0.5
2 每次reduce后都计算left个数，如果left个数 < target：中位数在right中，否则在left中
2 target更新规则：left区间target不变，right区间为 target - left
（target - left）if (target - left) > 0 else 新区间大小 + (target - left)
4 边界条件有些复杂，大概有5种情况
5 实际情况不需要

第20-194行用多进程实现了异构的master-worker分布式算法，并且数据只需要交换1次。
第200-368行用线程池模拟分布式算法，并且数据只需要交换1次，缺点是将Data放入了队列中，和实际情况不太符合。
第373行以后是原算法，缺点是数据需要交换多次。
"""


def median(lst, map_volume=500000, workers_number=2):
    master = Master(workers_number=workers_number, map_volume=map_volume)
    return master.run(lst)


class Master:
    def __init__(self, workers_number, map_volume):
        self.executor_pool = ProcessPoolExecutor(workers_number)
        self.workers_number = workers_number
        self.workers = []
        self.common_queue = Queue(maxsize=workers_number)
        self.single_queue = [Queue()] * workers_number
        self.single_confirm_queue = [Queue()] * workers_number
        self.map_volume = map_volume

    def confirm(self):
        for i in range(self.workers_number):
            self.single_confirm_queue[i].get()

    def first_count(self, lst, number):
        for i in range(self.workers_number):
            if i == self.workers_number - 1:
                _lst = lst[i * self.map_volume:]
            else:
                _lst = lst[i * self.map_volume:(i + 1) * self.map_volume]
            worker = Worker(lst=_lst, queue=self.single_queue[i], common_queue=self.common_queue,
                            confirm_queue=self.single_confirm_queue[i])
            worker.start()
            self.workers.append(worker)

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "do", "number": number})
        self.confirm()

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "count"})
        self.confirm()

        total_left, total_right = 0, 0
        for i in range(self.workers_number):
            left, right = self.common_queue.get()
            total_left += left
            total_right += right

        return total_left, total_right

    def _count(self, is_left):

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "rebuild", "is_left": is_left})
        self.confirm()

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "sample"})
        self.confirm()

        random_numbers = []
        for i in range(self.workers_number):
            _result = self.common_queue.get()
            if _result is not None:
                random_numbers.append(_result)
        number = random.sample(random_numbers, 1)[0]

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "do", "number": number})
        self.confirm()

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "count"})
        self.confirm()

        total_left, total_right = 0, 0
        for i in range(self.workers_number):
            message = self.common_queue.get()
            left, right = message
            total_left += left
            total_right += right
        return total_left, total_right

    def run(self, lst):
        first_number = random.sample(lst, 1)[0]
        total_left, total_right = self.first_count(lst, first_number)
        total = total_left + total_right
        target = (total_left + total_right) / 2 + 0.5
        while total > 1000:
            if total_left == target:
                return first_number
            elif total_left < target:
                target = target - total_left
                is_left = False
            else:
                is_left = True
            total_left, total_right = self._count(is_left)
            total = total_left + total_right

        for i in range(self.workers_number):
            self.single_queue[i].put({"type": "res"})
        self.confirm()
        res = []

        for i in range(self.workers_number):
            tmp = self.common_queue.get()
            res += tmp

        res = sorted(res)
        if target.is_integer():
            return res[int(target) - 1]
        else:
            return (res[int(target - 1.5)] + res[int(target - 0.5)]) / 2


class Worker(Process):
    def __init__(self, lst, queue, common_queue, confirm_queue):
        super().__init__()
        self.lst = lst
        self.left = []
        self.right = []
        self.queue = queue
        self.common_queue = common_queue
        self.confirm_queue = confirm_queue
        self.tmp_left_count = 0
        self.tmp_right_count = 0

    def count(self):
        self.common_queue.put((self.tmp_left_count, self.tmp_right_count))
        self.confirm_queue.put(1)

    def res(self):
        self.common_queue.put(self.lst)
        self.confirm_queue.put(1)

    def do(self, number):
        left_count, right_count = 0, 0
        for i in self.lst:
            if i <= number:
                left_count += 1
                self.left.append(i)
            else:
                right_count += 1
                self.right.append(i)
        self.tmp_left_count = left_count
        self.tmp_right_count = right_count
        self.confirm_queue.put(1)

    def rebuild(self, is_left):
        if is_left:
            self.lst = self.left
            self.tmp_right_count = 0
        else:
            self.lst = self.right
            self.tmp_left_count = 0
        self.left = []
        self.right = []
        self.confirm_queue.put(1)

    def sample(self):
        self.common_queue.put(
            random.sample(self.lst, 1)[0] if (self.tmp_left_count + self.tmp_right_count) > 0 else None)
        self.confirm_queue.put(1)

    def run(self):
        while True:
            data = self.queue.get()
            if data["type"] == "do":
                self.do(data["number"])
            if data["type"] == "count":
                self.count()
            if data["type"] == "rebuild":
                self.rebuild(data["is_left"])
            if data["type"] == "sample":
                self.sample()
            if data["type"] == "res":
                self.res()
            if data["type"] == "close":
                self.close()


"""
以下是用线程池模拟的分布式算法
"""
# class Master:
#     def __init__(self, workers_number, map_volume):
#         self.executor_pool = ThreadPoolExecutor(workers_number)
#         self.workers_number = workers_number
#         self.workers_data = Queue()
#         self.message_queue = Queue()
#         self.map_volume = map_volume
#
#     def first_count(self, lst, number):
#         for i in range(self.workers_number):
#             if i == self.workers_number - 1:
#                 self.workers_data.put(self.executor_pool.submit(Data, lst[i * self.map_volume:]))
#             else:
#                 self.workers_data.put(
#                     self.executor_pool.submit(Data, lst[i * self.map_volume:(i + 1) * self.map_volume]))
#
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             future = self.executor_pool.submit(Worker.run, data, number)
#             self.workers_data.put(future)
#
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             future = self.executor_pool.submit(Worker.count, data)
#             self.message_queue.put(future)
#             future = self.executor_pool.submit(Worker.put, data)
#             self.workers_data.put(future)
#
#         total_left, total_right = 0, 0
#         for i in range(self.workers_number):
#             left, right = self.message_queue.get().result()
#             total_left += left
#             total_right += right
#
#         return total_left, total_right
#
#     def _count(self, is_left):
#
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             future = self.executor_pool.submit(Worker.rebuild, data, is_left=is_left)
#             self.workers_data.put(future)
#
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             future = self.executor_pool.submit(Worker.sample, data)
#             self.message_queue.put(future)
#             future = self.executor_pool.submit(Worker.put, data)
#             self.workers_data.put(future)
#
#         random_numbers = []
#         for i in range(self.workers_number):
#             _result = self.message_queue.get().result()
#             if _result is not None:
#                 random_numbers.append(_result)
#         number = random.sample(random_numbers, 1)[0]
#
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             future = self.executor_pool.submit(Worker.run, data, number)
#             self.workers_data.put(future)
#
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             message_future = self.executor_pool.submit(Worker.count, data)
#             future = self.executor_pool.submit(Worker.put, data)
#             self.workers_data.put(future)
#             self.message_queue.put(message_future)
#
#         total_left, total_right = 0, 0
#         for i in range(self.workers_number):
#             message = self.message_queue.get().result()
#             left, right = message
#             total_left += left
#             total_right += right
#         return total_left, total_right
#
#     def run(self, lst):
#         first_number = random.sample(lst, 1)[0]
#         total_left, total_right = self.first_count(lst, first_number)
#         total = total_left + total_right
#         target = (total_left + total_right) / 2 + 0.5
#         while total > 1000:
#             if total_left == target:
#                 return first_number
#             elif total_left < target:
#                 target = target - total_left
#                 is_left = False
#             else:
#                 is_left = True
#             total_left, total_right = self._count(is_left)
#             total = total_left + total_right
#
#         futures = []
#         for i in range(self.workers_number):
#             data = self.workers_data.get().result()
#             futures.append(self.executor_pool.submit(Worker.res, data))
#         res = []
#         for future in futures:
#             tmp = future.result()
#             res += tmp
#         res = sorted(res)
#         if target.is_integer():
#             return res[int(target) - 1]
#         else:
#             print(target)
#             return (res[int(target - 1.5)] + res[int(target - 0.5)]) / 2
#
#
# class Worker:
#
#     @staticmethod
#     def run(cls, number):
#         left_count, right_count = 0, 0
#         for i in cls.lst:
#             if i <= number:
#                 left_count += 1
#                 cls.left.append(i)
#             else:
#                 right_count += 1
#                 cls.right.append(i)
#         cls.tmp_left_count = left_count
#         cls.tmp_right_count = right_count
#         return cls
#
#     @staticmethod
#     def count(cls):
#         return cls.count
#
#     @staticmethod
#     def res(cls):
#         return cls.res
#
#     @staticmethod
#     def rebuild(cls, is_left):
#         if is_left:
#             cls.lst = cls.left
#             cls.tmp_right_count = 0
#         else:
#             cls.lst = cls.right
#             cls.tmp_left_count = 0
#         cls.left = []
#         cls.right = []
#         return cls
#
#     @staticmethod
#     def sample(cls):
#         return random.sample(cls.lst, 1)[0] if (cls.tmp_left_count + cls.tmp_right_count) > 0 else None
#
#     @staticmethod
#     def put(cls):
#         return cls
#
#
# class Data:
#     def __init__(self, lst):
#         self.lst = lst
#         self.left = []
#         self.right = []
#         self.tmp_left_count = 0
#         self.tmp_right_count = 0
#
#     @property
#     def count(self):
#         return self.tmp_left_count, self.tmp_right_count
#
#     @property
#     def res(self):
#         return self.lst

"""
以下是原算法
"""
# def _map(l, m):
#     left = []
#     right = []
#     count_left = 0
#     for i in l:
#         if i <= m:
#             left.append(i)
#             count_left += 1
#         else:
#             right.append(i)
#     return count_left, left, right
#
#
# def _first_map(l, m):
#     left = []
#     right = []
#     count_left = 0
#     count_right = 0
#     for i in l:
#         if i <= m:
#             left.append(i)
#             count_left += 1
#         else:
#             right.append(i)
#             count_right += 1
#     return count_left, count_right, left, right
#
#
# def median(lst):
#     workers = 2
#     _map_number = 500000
#     executor = ProcessPoolExecutor(workers)
#
#     def _reduce(_lst=[], is_first=False, target=0.0):
#         if len(_lst) == 1:
#             executor.shutdown()
#             return _lst[0]
#         if len(_lst) == 2:
#             executor.shutdown()
#             if target.is_integer():
#                 return min(_lst)
#             return (_lst[0] + _lst[1]) / 2
#
#         lefts, rights = [], []
#         _m = random.sample(_lst, 1)[0]
#         if is_first:
#             total_left = 0
#             total_right = 0
#             futures = []
#             for i in range(workers):
#                 if i == workers - 1:
#                     futures.append(executor.submit(_first_map, l=_lst[i * _map_number:], m=_m))
#                 else:
#                     futures.append(
#                         executor.submit(_first_map, l=_lst[i * _map_number:(i + 1) * _map_number], m=_m))
#             for future in futures:
#                 count_left, count_right, left, right = future.result()
#                 total_left += count_left
#                 total_right += count_right
#                 lefts += left
#                 rights += right
#             total = total_left + total_right
#             target = total / 2 + 0.5
#         else:
#             total_left = 0
#             futures = []
#             for i in range(workers):
#                 if i == workers - 1:
#                     futures.append(executor.submit(_map, l=_lst[i * _map_number:], m=_m))
#                 else:
#                     futures.append(executor.submit(_map, l=_lst[i * _map_number:(i + 1) * _map_number], m=_m))
#             for future in futures:
#                 count_left, left, right = future.result()
#                 lefts += left
#                 rights += right
#                 total_left += count_left
#
#         if total_left == target and target.is_integer():
#             return _m
#         elif total_left < target:
#             target = target - total_left
#             if target == 0.5:
#                 return (max(lefts) + min(rights)) / 2
#             return _reduce(_lst=rights, target=target)
#         else:
#             if target == 0.5:
#                 return (max(lefts) + min(rights)) / 2
#             return _reduce(_lst=lefts, target=target)
#
#     return _reduce(_lst=lst, is_first=True)
