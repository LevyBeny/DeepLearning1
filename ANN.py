import numpy as np


# initialize the W and b parameters for the network model for each layer
def initialize_parameters(layer_dims):
    network_params={}

    #  W=[prev_layer_dim,current_layer_dim], b=[current_layer_dim]
    for i in range(1, len(layer_dims)):
        network_params[i] = {}
        network_params[i]["W"]=(np.random.randn(layer_dims[i-1],layer_dims[i]))
        network_params[i]["b"]=(np.zeros(layer_dims[i],1))

    return network_params

#
def linear_forward(A, W, b):
    Z=np.transpose(W)@A+b
    linear_cache = {"A_prev":A,"W":W,"b":b,"Z":Z}
    return Z, linear_cache


def sigmoid(Z):
    return 1/(1+np.exp(-Z)),Z


def relu(Z):
    return np.maximum(Z,np.zeros(Z.shape)) , Z


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev , W, b)
    if activation == "relu":
        A = relu(Z)
    if activation == "sigmoid":
        A = sigmoid(Z)
    else:
        pass
    return A, linear_cache


def L_model_forward(X, parameters):
    A=X
    caches_list=[]
    activation = "relu"

    for i in range(len(parameters)):
        if i == len(parameters)-1:
            activation = "sigmoid"
        W, b = X[i][0],X[i][1]
        A, linear_cache = linear_activation_forward(A, W, b, activation)
        caches_list.append(linear_cache)

    return A, caches_list

def compute_cost(AL, Y):
    m = float(Y.shape[1])
    y = Y.reshape(m)
    al = AL.reshape(m)
    toSum = y * np.log(al) + ((1-y) * np.log((1-al)))
    cost = -1/m * np.sum(toSum)
    return cost


##### BACK PROPOGATION #####

def linear_backward( dZ, cache):
    m= cache["A_prev"].shape[1]

    dA_prev = np.transpose(cache["W"]) @ dZ
    dW = (1/m)*(dZ) @ np.transpose(cache["A_prev"])
    db =(1/m)* np.sum(dZ, axis =1,keepdims=True)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    if activation == "relu":
        dZ = relu_backward(dA, cache)
    else:
        dZ = sigmoid_backward(dA, cache)

    return linear_backward(dZ, cache)

def relu_backward (dA, activation_cache):
    if activation_cache["Z"] < 0:
        return 0
    return dA

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache['Z']
    res = 1 / (1 + np.exp(-Z))
    dZ = dA * res * (1 - res)
    return dZ

#
def L_model_backward(AL, Y, caches):
    grads={}
    dAL = -(np.divide(Y, AL)-np.divide(1-Y, 1-AL))

    # last layer
    dA, dW, db = linear_activation_backward(dAL, caches[len(caches)-1], "sigmoid")
    grads["dA"+str(len(caches)-1)] = dA
    grads["dW" + str(len(caches) - 1)] = dW
    grads["db" + str(len(caches) - 1)] = db

    for layer in range(len(caches)-2, 0 , -1):
        dA, dW, db = linear_activation_backward(dA, caches[layer], "relu")
        grads["dA" + str(layer)] = dA
        grads["dW" + str(layer)] = dW
        grads["db" + str(layer)] = db

    return grads

def Update_parameters(parameters, grads, learning_rate):
    for i in range(len(parameters)):
        dW = grads["dW" + str(i)]
        db = grads["db" + str(i)]
        parameters[i]["W"] = parameters[i]["W"]-learning_rate*dW
        parameters[i]["b"] = parameters[i]["b"]-learning_rate*db

    return parameters