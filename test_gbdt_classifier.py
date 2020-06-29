from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from algs.ml_algorithms.gbdt import GbdtClassifier
from algs.utils import compute_confusion_matrix

x, y = load_breast_cancer(return_X_y=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

gbdt = GbdtClassifier(max_depth=5, min_leave_samples=5, learning_rate=0.1)
gbdt.fit(x_train, y_train)

y_predict = gbdt.predict(x_test)
print(compute_confusion_matrix(y_predict, y_test))
