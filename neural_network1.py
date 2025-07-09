import numpy as np

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


# now for the output taking these activations as the input to the output layer
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print("weighted sum of the neural network is = {}".format(np.around(z_2, decimals= 2)))

# applying the activation function on the output layer
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print("output of the neural network is = {}".format(np.around(z_2, decimals= 4)))

"""So this was the simple example but for the real problems we will have many hidden lavers and n number of inputs."""


# lets start building this neural network by taking 2 hidden layers:
n = 2
hidden_layers = 2
m = [2,2]  # number of nodes in each layer
output = 1  # number of output nodes

previous = n # nodes in the previous layer
network = {}

# looping over the layer and randomly initializing the weights and biases
for layer in range(hidden_layers + 1):
    if layer == hidden_layers:
        name = "output layer"
        num_nodes = output
    else:
        name = "layer_{}".format(layer +1)
        num_nodes = m[layer]

    network[name] = {}
    for node in range(num_nodes):
        node_name = "node_{}".format(node +1)
        network[name][node_name] = {
            'weights': np.around(np.random.uniform(size=previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }
    previous = num_nodes
    
print(network)

