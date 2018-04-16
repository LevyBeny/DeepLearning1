import numpy as np


# initialize the W and b parameters for the network model for each layer
def initialize_parameters(layer_dims):
    network_params={}

    #  W=[prev_layer_dim,current_layer_dim], b=[current_layer_dim]
    for i in range(1, len(layer_dims)):
        network_params[i] ={}
        network_params[i]["W"] = np.random.randn(layer_dims[i-1],layer_dims[i])
        network_params[i]["b"] = np.zeros(layer_dims[i],1)

    return network_params

#
def linear_forward(A, W, b):
    Z=np.transpose(W)*A+b
    linear_cache = {"A":A,"W":W,"b":b,"Z":Z}
    return Z, linear_cache


def sigmoid(Z):
    return 1/(1+np.exp(-Z)),Z


def relu(Z):
    return np.maximum(Z,np.zeros(Z.shape)) , Z


def linear_activation_forward(A_prev, W, B, activation):
    pass


def L_model_forward(X, parameters):
    pass

def compute_cost(AL, y)::
    pass