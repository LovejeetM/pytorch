import numpy as np
from random import seed

# 6 weights
# 3 biases

weights  = np.around(np.random.uniform(size = 6), decimals = 2)
biases = np.around(np.random.uniform(size = 3), decimals = 2)

print(weights)
print(biases)

# now lets take inputs to the neural network
x1 = 0.50
x2 = 0.85

print("x1 = {}, x2 = {}".format(x1, x2))

# computing the weighted sum at the first hidden layer z(1,1)
z_11 = x1 * weights[0] + x2 * weights[1] + biases[0]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(np.around(z_11, decimals = 2)))

# computing the weighted sum at the first hidden layer z(1,2)
z_12 = x1 * weights[2] + x2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals = 2)))

# applying the activation function (sigmoid) in the hidden layer on first node
a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

# applying the activation function (sigmoid) in the hidden layer on second node
a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))


# now for the num_nodes_output taking these activations as the input to the output layer
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print("weighted sum of the neural network is = {}".format(np.around(z_2, decimals= 2)))

# applying the activation function on the output layer
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print("output of the neural network is = {}".format(np.around(z_2, decimals= 4)))

"""So this was the simple example but for the real problems we will have many hidden lavers and n number of inputs."""


# lets start building this neural network by taking 2 hidden layers:
n = 2
num_hidden_layers = 2
m = [2,2]  # number of nodes in each layer
num_nodes_output = 1  # number of output nodes

previous = n # nodes in the num_nodes_previousprevious layer
network = {}

# looping over the layer and randomly initializing the weights and biases
for layer in range(num_hidden_layers + 1):
    if layer == num_hidden_layers:
        layer_name = "output layer"
        num_nodes = num_nodes_output
    else:
        layer_name = "layer_{}".format(layer +1)
        num_nodes = m[layer]

    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = "node_{}".format(node +1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }
    previous = num_nodes
    
print(network)

# Creating a function to call this to initialize the network
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = "output layer"
            num_nodes = num_nodes_output
        else:
            layer_name = "layer_{}".format(layer +1)
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = "node_{}".format(node +1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    
    return network

s = initialize_network(5, 3, [3, 2, 3], 1)
print(s)

def compute_weighted_sum(inputs, weights, biases):
    return np.sum(inputs * weights) + biases

np.random.seed(12)
input1 = np.around(np.random.uniform(size=5), decimals=2)

print('The inputs to the network are {}'.format(input1))

print(s['layer_1']['node_1']['weights'])
print(s['layer_1']['node_1']['bias'][0])

print(compute_weighted_sum(input1, s['layer_1']['node_1']['weights'], s['layer_1']['node_1']['bias'][0]))
a = compute_weighted_sum(input1, s['layer_1']['node_1']['weights'], s['layer_1']['node_1']['bias'][0])

def node_activation(weighted_sum):
    return (1.0 / (1.0 + np.exp(-1 * weighted_sum)))

ws = node_activation(a)
print("Node activation  = {}".format(np.around(ws, decimals = 4)))



def forward_propagate(network, inputs):
    layer_inputs = list(inputs)

    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals= 4))

        if layer != "output layer":
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs

    network_predictions = layer_outputs
    return network_predictions

print(forward_propagate(s, input1))


# Creating network, then giving inputs and then making predictions by using this function.
network1 = initialize_network(5, 3, [2, 3, 2], 3)
input2 = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(network1, input2)
print('The predicted values by the network for the given input are {}'.format(predictions))
