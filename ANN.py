import numpy as np


# initialize the W and b parameters for the network model for each layer
def initialize_parameters(layer_dims):
    network_params = []
    #  W=[prev_layer_dim,current_layer_dim], b=[current_layer_dim]
    for i in range(1, len(layer_dims)):
        layer_params = {"W": (np.random.randn(layer_dims[i], layer_dims[i-1]))*0.05,
                        "b": (np.zeros(layer_dims[i])).reshape(layer_dims[i], 1)}
        network_params.append(layer_params)

    return network_params


# The computation of the linear part of the forward propagation
def linear_forward(A, W, b):
    Z = W @ A + b
    linear_cache = {"A_prev": A, "W": W, "b": b, "Z": Z}
    return Z, linear_cache


# The sigmoid forward propagation calculation
def sigmoid(Z):
    return np.nan_to_num(1 / (1 + np.exp(-Z))), Z


# The relu forward propagation calculation
def relu(Z):
    return np.maximum(Z, np.zeros(Z.shape)), Z


# Calculates both linear and activation for forward propagation using the linear and activation functions.
def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, _ = relu(Z)
    if activation == "sigmoid":
        A, _ = sigmoid(Z)
    else:
        pass
    return A, linear_cache


# Applies the forward propagation process
def L_model_forward(X, parameters):
    A = X
    caches_list = []
    activation = "relu"

    for i in range(len(parameters)):
        if i == len(parameters) - 1:
            activation = "sigmoid"
        W, b = parameters[i]["W"], parameters[i]["b"]
        A, linear_cache = linear_activation_forward(A, W, b, activation)
        caches_list.append(linear_cache)

    return A, caches_list


# computes the cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    y = Y.reshape(m)
    al = AL.reshape(m)
    toSum = np.nan_to_num( y * np.log(al) + ((1 - y) * np.log((1 - al))))
    cost = -1 / float(m) * np.sum(toSum)
    return cost


##### BACK PROPOGATION #####

# The computation of the linear part of the backward propagation
def linear_backward(dZ, cache):
    m = cache["A_prev"].shape[1]

    dA_prev = np.transpose(cache["W"]) @ dZ
    dW = (1 / m) * dZ @ np.transpose(cache["A_prev"])
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    return dA_prev, dW, db


# Calculates both linear and activation for forward propagation using the linear and activation functions.
def linear_activation_backward(dA, cache, activation):
    if activation == "relu":
        dZ = relu_backward(dA, cache)
    else:
        dZ = sigmoid_backward(dA, cache)

    return linear_backward(dZ, cache)


# The computation of the relu function backward propagation
def relu_backward(dA, activation_cache):
    dZ = np.array(dA, copy=True)
    dZ[activation_cache["Z"] <= 0] = 0
    return dZ


# The computation of the sigmoid function backward propagation
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache['Z']
    res = 1 / (1 + np.exp(-Z))
    dZ =np.nan_to_num( dA * res * (1 - res))
    return dZ


# The computation of the backward gradients
def L_model_backward(AL, Y, caches):
    grads = {}
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Last layer - sigmoid
    dA, dW, db = linear_activation_backward(dAL, caches[len(caches) - 1], "sigmoid")
    grads["dA" + str(len(caches) - 1)] = dA
    grads["dW" + str(len(caches) - 1)] = dW
    grads["db" + str(len(caches) - 1)] = db

    # Rest of the layers - relu
    for layer in range(len(caches) - 2, -1, -1):
        dA, dW, db = linear_activation_backward(dA, caches[layer], "relu")
        grads["dA" + str(layer)] = dA
        grads["dW" + str(layer)] = dW
        grads["db" + str(layer)] = db

    return grads


# Update the W and b parameters accortding to the gradients and the learning rate
def Update_parameters(parameters, grads, learning_rate):
    for i in range(len(parameters)):
        dW = grads["dW" + str(i)]
        db = grads["db" + str(i)]
        parameters[i]["W"] = parameters[i]["W"] - learning_rate * dW
        parameters[i]["b"] = parameters[i]["b"] - learning_rate * db

    return parameters


# Applies the whole training process on the network.
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    # initialize ->L_model_forward -> compute_cost -> L_model_backward -> update parameters

    costs = []
    layers_dims.insert(0, X.shape[0])
    initial_params = initialize_parameters(layers_dims)
    parameters = initial_params

    for i in range(1, num_iterations + 1):
        AL, caches_List = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if i % 100 == 0:
            costs.append(cost)
            print("cost - "+str(i)+":"+str(cost))
        grads = L_model_backward(AL, Y, caches_List)
        parameters=Update_parameters(initial_params, grads, learning_rate)

    return parameters, costs


# Predict the expected labels of the X data using the trained model. returns the model accuracy.
def Predict(X, Y, parameters):
    A, _ = L_model_forward(X, parameters)
    predict_y = [1 if prediction >= 0.5 else 0 for prediction in A.reshape(A.shape[1])]

    # Calculate Accuracy
    denominator = 0.0
    for i in range(Y.shape[1]):
        if Y[0,i] == predict_y[i]:
            denominator += 1

    return denominator / Y.shape[1]
