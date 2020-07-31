from functools import reduce

import dill


def save_model(model, file_path, clean=True):
    if clean:
        model.clean()
    res = dill.dumps(obj=model)
    with open(file_path, 'wb') as f:
        f.write(res)


def load_model(file_path):
    res = bytes("", encoding='utf-8')
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        res += reduce(lambda x, y: x + y, lines)
    return dill.loads(res)


def load_model(file_path):
    f = open(file_path, "rb")
    return dill.load(f)
