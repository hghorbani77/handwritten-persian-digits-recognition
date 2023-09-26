import numpy as np
import glob as gb
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.utils import plot_model

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

n_inputs = X.shape[1]  # Number of inputs

# Scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

# Define Encoder
visible = Input(shape=(n_inputs,))

# level 1
e = Dense(n_inputs * 2)(visible)  # e = encoder
e = BatchNormalization()(e)
e = LeakyReLU()(e)

# level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

# Bottleneck
n_bottleneck = round(float(n_inputs) / 2.0)
bottleneck = Dense(n_bottleneck)(e)

# Define Decoder
# level 1
d = Dense(n_inputs)(bottleneck)  # d = de encoder
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# level 2
d = Dense(n_inputs * 2)(d)  # d = de encoder
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# Output layer
output = Dense(n_inputs, activation='linear')(d)

# Define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder
model.compile(optimizer='adam', loss='mse')

# Fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=200, batch_size=16,
                    verbose=2, validation_data=(X_test, X_test))

# Plot Loss
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Autoencoder loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Autoencoder loss.png')

# Define an encoder model without the decoder
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(model, 'encoder_compress.png', show_shapes=True)

model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)

# encode the  data
X_train_encode = encoder.predict(X_train)
X_test_encode = encoder.predict(X_test)

model_encode = LogisticRegression()
# fit the model on the training set
model_encode.fit(X_train_encode, Y_train)
# make predictions on the test set
y_pred_encode = model_encode.predict(X_test_encode)
acc_encode = accuracy_score(Y_test, y_pred_encode)

print('Accuracy LogisticRegression Simple: {}'.format(acc))
print('Accuracy LogisticRegression With Autoencoder: {}'.format(acc_encode))

# data concatenation
X_train_enc = np.concatenate((X_train, X_train_encode), axis=1)
X_test_enc = np.concatenate((X_test, X_test_encode), axis=1)

mlp2 = MLPClassifier(hidden_layer_sizes=(80, 20), activation='tanh',
                     solver='sgd', batch_size=150, learning_rate='adaptive',
                     learning_rate_init=0.001, max_iter=300, shuffle=True,
                     tol=0.001, verbose=True, momentum=0.95)
# train neural network
mlp2.fit(X_train_enc, Y_train)

print('train accuracy :', mlp2.score(X_train_enc, Y_train))
print('test accuracy :', mlp2.score(X_test_enc, Y_test))

loss = mlp2.loss_curve_

plt.plot(loss, label='loss-train')
plt.title('MLP loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('Plots/MLP2 loss.png')

Y_test_pred = mlp2.predict(X_test_enc)
mlp_test_cm = cm(Y_test, Y_test_pred)

Y_train_pred = mlp2.predict(X_train_enc)
mlp_train_cm = cm(Y_train, Y_train_pred)

mlp2_report_test = classification_report(Y_test, Y_test_pred)
mlp2_report_train = classification_report(Y_train, Y_train_pred)
