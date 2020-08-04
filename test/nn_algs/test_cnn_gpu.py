import warnings
import cupy as np
from algs.nn_algoritms_gpu.common.model import Model
from algs.nn_algoritms_gpu.common.layer import Dense, Conv2d, Flatten
from tensorflow.keras.datasets import cifar10

warnings.filterwarnings("ignore")

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images = (train_images / 255).astype('float16').transpose([0, 3, 2, 1])
test_images = (test_images / 255).astype('float16').transpose([0, 3, 2, 1])

model = Model(lr=0.005, epoch=150, loss="Crossentropy_with_softmax", classes=10,
              optimizer='sgd_with_momentum', decay=0.999992,
              early_stop=True, tol=2e-4, momentum_beta=0.9, batch_size=8, shuffle=1)
model.add(Conv2d(activation='leakyrelu', units=32, kernel_size=[3, 3], strides=1, padding='valid'))
model.add(Conv2d(activation='leakyrelu', units=64, kernel_size=[3, 3], strides=1, padding='valid'))
model.add(Conv2d(activation='leakyrelu', units=128, kernel_size=[3, 3], strides=1, padding='valid'))

model.add(Flatten(input_shape=4))
model.add(Dense(activation='leakyrelu', units=256))
model.add(Dense(activation='softmax', units=10))
model.fit(train_images, train_labels, watch_loss=1)

from algs.nn_algoritms_gpu.common.utils import save_model

save_model(model, "./test.m")
