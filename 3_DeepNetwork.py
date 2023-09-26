import numpy as np
import glob as gb
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import AveragePooling2D
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot

# loading data

images_path = gb.glob('Hoda_Dataset/' + "*.bmp")

# feature extraction
X = []
Y = []

for i in range(len(images_path)):
    path = images_path[i]
    idx = path.rfind('_') - 1
    Y.append(int(path[idx]))
    img = cv2.imread(path, 0)
    img = cv2.resize(img, dsize=(28, 28))
    X.append(img)

X = np.array(X)
Y = np.array(Y)

# shuffle data
list_per = np.random.permutation(len(images_path))
X_per = []
Y_per = []
for i in range(len(list_per)):
    idx_per = list_per[i]
    ftr = X[idx_per]
    lbl = Y[idx_per]
    X_per.append(ftr)
    Y_per.append(lbl)
X_per = np.array(X_per)
Y_per = np.array(Y_per)

# splitting data to train and test
train_test_split = int(0.7 * len(images_path))
X_train = X_per[0:train_test_split, :]
Y_train = Y_per[0:train_test_split]

X_test = X_per[train_test_split:, :]
Y_test = Y_per[train_test_split:]

y_train = keras.utils.to_categorical(Y_train, 10)
y_test = keras.utils.to_categorical(Y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255
X_test = X_test / 255

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

batch_size = 150
num_classes = 10
epochs = 20

model = Sequential()
model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(16, 5, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train, epochs=20, validation_split=0.2)
print("The model has successfully trained")

score_train = model.evaluate(X_train, y_train, verbose=0)
score_test = model.evaluate(X_test, y_test, verbose=0)

print('Train accuracy:', score_train[1])
print('Test accuracy:', score_test[1])

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig("Plots/CNN plots.png")

