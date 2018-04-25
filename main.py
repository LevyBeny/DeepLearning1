from keras.datasets import mnist
import ANN
import numpy as np

# extract data by given digits
def get_data_by_digits(X,y,digit1,digit2):
    indices = np.where(np.logical_or(y_train == digit1, y_train == digit2))
    X_new = X_train[:, indices[0]]
    y_new = y_train[indices[0]].reshape(1, -1)
    y_new=(y_new == digit1).astype(np.float32)
    return X_new,y_new

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten the data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]).T
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).T
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# extract only 3 & 8 digits
X_train_38,y_train_38 = get_data_by_digits(X_train,y_train,3,8)
X_test_38,y_test_38=get_data_by_digits(X_test,y_test,3,8)

# extract only 7 & 9 digits
X_train_79,y_train_79 = get_data_by_digits(X_train,y_train,7,9)
X_test_79,y_test_79=get_data_by_digits(X_test,y_test,7,9)

parameters, costs = ANN.L_layer_model(X_train_38, y_train_38, layers_dims=[20, 7, 5, 1], learning_rate=0.009,
                                      num_iterations=3000)
acc=ANN.Predict(X_test_38,y_test_38,parameters)
print(acc)
print("Acc:"+str(acc)+"\n")
print("parameters:")
print(parameters)
print("costs:")
print(costs)