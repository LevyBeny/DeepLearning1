import numpy as np


# initialize the W and b parameters for the network model for each layer
def initialize_parameters(layer_dims):
    network_params = {}

    for i in range(1, len(layer_dims)):
        network_params['W' + str(i)] = (np.random.randn(layer_dims[i], layer_dims[i - 1])) * 0.05
        network_params['b' + str(i)] = (np.zeros(layer_dims[i])).reshape(layer_dims[i], 1)

    return network_params


# The computation of the linear part of the forward propagation
def linear_forward(A, W, b):
    Z = W @ A + b
    linear_cache = {"A_prev": A, "W": W, "b": b}
    return Z, linear_cache


# The sigmoid forward propagation calculation
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z


# The relu forward propagation calculation
def relu(Z):
    return np.maximum(Z, np.zeros(Z.shape)), Z


# Calculates both linear and activation for forward propagation using the linear and activation functions.
def linear_activation_forward(A_prev, W, b, activation):
    cache = {}
    Z, cache['linear_cache'] = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, cache['activation_cache'] = relu(Z)
    elif activation == "sigmoid":
        A, cache['activation_cache'] = sigmoid(Z)

    return A, cache


# Applies the forward propagation process
def L_model_forward(X, parameters):
    A = X
    caches_list = []
    activation = "relu"
    num_of_layers = int(len(parameters.keys())/2)
    for i in range(1, num_of_layers + 1):
        if i == num_of_layers:
            activation = "sigmoid"
        W, b = parameters["W" + str(i)], parameters["b" + str(i)]
        A, cache = linear_activation_forward(A, W, b, activation)
        caches_list.append(cache)

    return A, caches_list


# computes the cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    y = Y.reshape(m)
    al = AL.reshape(m)
    toSum = np.nan_to_num(y * np.log(al) + ((1 - y) * np.log((1 - al))))
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
        dZ = relu_backward(dA, cache['activation_cache'])
    else:
        dZ = sigmoid_backward(dA, cache['activation_cache'])

    return linear_backward(dZ, cache['linear_cache'])


# The computation of the relu function backward propagation
def relu_backward(dA, activation_cache):
    dZ = np.array(dA, copy=True)
    dZ[activation_cache <= 0] = 0
    return dZ


# The computation of the sigmoid function backward propagation
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    res = 1 / (1 + np.exp(-Z))
    dZ = dA * res * (1 - res)
    return dZ


# The computation of the backward gradients
def L_model_backward(AL, Y, caches):
    grads = {}
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Last layer - sigmoid
    dA, dW, db = linear_activation_backward(dAL, caches[- 1], "sigmoid")
    grads["dA" + str(len(caches))] = dA
    grads["dW" + str(len(caches))] = dW
    grads["db" + str(len(caches))] = db

    # Rest of the layers - relu
    for layer in range(len(caches) - 1, 0, -1):
        dA, dW, db = linear_activation_backward(dA, caches[layer-1], "relu")
        grads["dA" + str(layer)] = dA
        grads["dW" + str(layer)] = dW
        grads["db" + str(layer)] = db

    return grads


# Update the W and b parameters accortding to the gradients and the learning rate
def Update_parameters(parameters, grads, learning_rate):
    num_of_layers = int(len(parameters.keys())/2)
    for i in range(1, num_of_layers):
        dW = grads["dW" + str(i)]
        db = grads["db" + str(i)]
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * dW
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * db

    return parameters


# Applies the whole training process on the network.
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    # initialize ->L_model_forward -> compute_cost -> L_model_backward -> update parameters

    costs = []
    layers_dims.insert(0, X.shape[0])
    parameters = initialize_parameters(layers_dims)

    for i in range(1, num_iterations + 1):
        AL, caches_List = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if i % 100 == 0:
            costs.append(cost)
            print("cost - " + str(i) + ":" + str(cost))
        grads = L_model_backward(AL, Y, caches_List)
        parameters = Update_parameters(parameters, grads, learning_rate)

    return parameters, costs


# Predict the expected labels of the X data using the trained model. returns the model accuracy.
def Predict(X, Y, parameters):
    A, _ = L_model_forward(X, parameters)
    predict_y = [1.0 if prediction >= 0.5 else 0.0 for prediction in A.reshape(A.shape[1])]

    # Calculate Accuracy
    denominator = 0.0
    for i in range(Y.shape[1]):
        if Y[0, i] == predict_y[i]:
            denominator += 1

    return denominator / Y.shape[1]
