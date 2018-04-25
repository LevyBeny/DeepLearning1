from keras.datasets import mnist
import ANN
import numpy as np


# extract data by given digits
def get_data_by_digits(X,y,digit1,digit2):
    indices = np.where(np.logical_or(y == digit1, y == digit2))
    X_new = X[:, indices[0]]
    y_new = y[indices[0]].reshape(1, -1)
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

run_38 = {}
run_79 = {}
run_38["parameters"], run_38["costs"] = ANN.L_layer_model(X_train_38, y_train_38, layers_dims=[20, 7, 5, 1], learning_rate=0.009,
                                                          num_iterations=3000)

run_79["parameters"], run_79["costs"] = ANN.L_layer_model(X_train_79, y_train_79, layers_dims=[20, 7, 5, 1], learning_rate=0.009,
                                                          num_iterations=3000)

run_38["train_acc"]=ANN.Predict(X_train_38,y_train_38, run_38["parameters"])
run_38["test_acc"]=ANN.Predict(X_test_38,y_test_38, run_38["parameters"])

run_79["train_acc"]=ANN.Predict(X_train_79,y_train_79, run_79["parameters"])
run_79["test_acc"]=ANN.Predict(X_test_79,y_test_79, run_79["parameters"])

print("Run 7,9 labels test set Acc: "+str(run_79["test_acc"]) +"\n" )
print("Run 7,9 labels train set Acc: "+str(run_79["train_acc"]) +"\n" )
print("Run 3,8 labels test set Acc: "+str(run_38["test_acc"]) +" \n" )
print("Run 3,8 labels train set Acc: "+str(run_38["train_acc"]) +"\n" )

with open("cost_result.csv",mode="w") as f:
    f.write("index,")
    for i in range(len(run_79["costs"])):
        f.write(str(i)+",")
    f.write("\n")

    f.write("classify 7_9,")
    for cost in run_79["costs"]:
        f.write(str(cost)+",")
    f.write("\n")

    f.write("classify 3_8,")
    for cost in run_38["costs"]:
        f.write(str(cost)+",")
    f.write("\n")
# File 1 :
# Costs for the 2 settings : 3,8 and 7,9 -
# Cost : 3,8 - 30 costs
# Cost : 7,9 - 30 costs

