import warnings
from algs.nn_algoritms.common.model import Model
from algs.nn_algoritms.common.layer import Dense, Conv2d, Flatten
from tensorflow.keras.datasets import cifar10
warnings.filterwarnings("ignore")
1

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = (train_images / 255).astype('float16').transpose([0, 3, 2, 1])
test_images = (test_images / 255).astype('float16').transpose([0, 3, 2, 1])

model = Model(lr=0.1, epoch=250, loss="Crossentropy_with_softmax", classes=10,
              optimizer='sgd_with_momentum', decay=0.9999,
              early_stop=True, tol=2e-4, momentum_beta=0.9, batch_size=32, shuffle=1)
model.add(Conv2d(activation='relu', units=16, kernel_size=[3, 3], strides=1, padding='valid'))
model.add(Conv2d(activation='relu', units=32, kernel_size=[3, 3], strides=1, padding='valid'))
model.add(Flatten(input_shape=4))
model.add(Dense(activation='linear', units=10))
model.fit(train_images, train_labels, watch_loss=1)
