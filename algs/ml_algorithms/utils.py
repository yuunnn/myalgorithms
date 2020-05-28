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
    return pd.crosstab(ypred,yture)


def standardization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma
