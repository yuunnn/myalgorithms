from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from algs.ml_algorithms.cart import CartClassifier
from algs.utils import compute_confusion_matrix

x, y = load_iris(return_X_y=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

cart = CartClassifier(max_depth=5, min_leave_samples=5)
cart.fit(x_train, y_train)

y_predict = cart.predict(x_test)
print(compute_confusion_matrix(y_predict, y_test))
