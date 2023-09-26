import numpy as np
import glob as gb
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report

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
    img = cv2.resize(img, dsize=(10, 10))
    features = np.reshape(img, newshape=100)
    X.append(features)

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

# normalize
X_train = X_train / 255
X_test = X_test / 255

# create neural network
mlp = MLPClassifier(hidden_layer_sizes=(80, 20), activation='tanh',
                    solver='sgd', batch_size=150, learning_rate='adaptive',
                    learning_rate_init=0.001, max_iter=300, shuffle=True,
                    tol=0.001, verbose=True, momentum=0.95)
# train neural network
mlp.fit(X_train, Y_train)

print('train accuracy :', mlp.score(X_train, Y_train))
print('test accuracy :', mlp.score(X_test, Y_test))

loss = mlp.loss_curve_

plt.plot(loss, label='loss-train')
plt.title('MLP loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('Plots/MLP loss.png')

Y_test_pred = mlp.predict(X_test)
mlp_test_cm = cm(Y_test, Y_test_pred)

Y_train_pred = mlp.predict(X_train)
mlp_train_cm = cm(Y_train, Y_train_pred)

mlp_report_test = classification_report(Y_test, Y_test_pred)
mlp_report_train = classification_report(Y_train, Y_train_pred)
