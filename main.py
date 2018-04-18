from keras.datasets import mnist
import ANN
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]).T
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
print(y_train.shape)
# extract only 3 & 8 digits
y_train,indices =np.where( (y_train==8) | (y_train==3))
X_train_38=2

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).T
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# indices = np.ix_(train_y[])
