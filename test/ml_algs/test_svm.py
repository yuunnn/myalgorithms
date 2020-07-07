import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from algs.ml_algorithms.svm import Svm

from algs.utils import standardization

x, y = load_breast_cancer(return_X_y=1)
x = standardization(x)
y = np.array([-1 if i <= 0 else 1 for i in y])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

svm = Svm(max_iter=50, kernel='linear', tol=1e-5)
svm.fit(x_train, y_train)
ypre = svm.predict(x_test)
print(pd.crosstab(y_test, ypre))

