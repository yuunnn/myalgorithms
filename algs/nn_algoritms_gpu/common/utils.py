from functools import reduce

import dill


def save_model(model, file_path):
    res = dill.dumps(obj=model)
    with open(file_path, 'wb') as f:
        f.write(res)


def load_model(file_path):
    res = bytes("", encoding='utf-8')
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        res += reduce(lambda x, y: x + y, lines)
    return dill.loads(res)
