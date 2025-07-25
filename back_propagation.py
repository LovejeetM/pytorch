import numpy as np 
import matplotlib.pyplot as plt

x =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T        #input matrix
d = np.array([0, 1, 1, 0])            #expected output


def initialize_network_parameters():

    inputSize = 2      # Number of input neurons 
    hiddenSize = 2     # Number of hidden neurons
    outputSize = 1     # Number of output neurons
    lr = 0.01           # Learning rate
    epochs = 1000000    # Number of training epochs

    # Initialize weights and biases randomly within the range [-1, 1]
    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # Weights from input to hidden layer
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1          # Bias for hidden layer
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1 # Weights from hidden to output layer
    b2 = np.random.rand(outputSize, 1) * 2 - 1          # Bias for output layer

    return w1, b1, w2, b2, lr, epochs

w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

error_list = []
for epoch in range(epochs):
    z1 = np.dot(w1, x) + b1
    a1 = 1 / (1 + np.exp(-z1))  

    z2 = np.dot(w2, a1) + b2  
    a2 = 1 / (1 + np.exp(-z2))  

    error = d - a2  
    da2 = error * (a2 * (1 - a2))  
    dz2 = da2  

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * (a1 * (1 - a1))

    w2 += lr * np.dot(dz2, a1.T)
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)  

    w1 += lr * np.dot(dz1, x.T)  
    b1 += lr * np.sum(dz1, axis=1, keepdims=True) 
    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))

z1 = np.dot(w1, x) + b1  
a1 = 1 / (1 + np.exp(-z1))

z2 = np.dot(w2, a1) + b2
a2 = 1 / (1 + np.exp(-z2))

print('Final output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))

plt.plot(error_list)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  
d = np.array([0, 0, 0, 1])  

w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

error_list = []
for epoch in range(epochs):
    z1 = np.dot(w1, X) + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.dot(w2, a1) + b2  
    a2 = 1 / (1 + np.exp(-z2))

    error = d - a2
    da2 = error * (a2 * (1 - a2))
    dz2 = da2

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * (a1 * (1 - a1))

    w2 += lr * np.dot(dz2, a1.T)
    b2 += lr * np.sum(dz2, axis=1, keepdims=True) 

    w1 += lr * np.dot(dz1, X.T)
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)
    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))

z1 = np.dot(w1, X) + b1
a1 = 1 / (1 + np.exp(-z1))

z2 = np.dot(w2, a1) + b2
a2 = 1 / (1 + np.exp(-z2))

print('\nFinal output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))

plt.plot(error_list)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

"""  
Output:

Epoch: 9999, Average error: 0.49935
Epoch: 499999, Average error: 0.01721
Epoch: 999999, Average error: 0.01157
Final output after training: [[0.01157857 0.98910297 0.98666165 0.01045059]]
Ground truth [0 1 1 0]
Error after training: [[-0.01157858  0.01089704  0.01333836 -0.01045059]]
Average error: 0.01157

Epoch: 9999, Average error: 0.35898
Epoch: 499999, Average error: 0.00908
Epoch: 999999, Average error: 0.00609
Final output after training: [[5.90469779e-04 7.01595300e-03 7.18880465e-03 9.90420412e-01]]
Ground truth [0 0 0 1]
Error after training: [[-0.00059047 -0.00701596 -0.00718881  0.00957959]]
Average error: 0.00609

"""