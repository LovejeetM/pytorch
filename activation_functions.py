import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z >0, 1, 0)

z = np.linspace(-10, 10, 400)

# sigmoid_grad = [sigmoid_derivative(i) for i in z]
# relu_grad = [relu_derivative(i) for i in z]

sigmoid_grad = sigmoid_derivative(z)
relu_grad = relu_derivative(z)

print(sigmoid_grad)
print(relu_grad)

plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

z = np.linspace(-5, 5, 100)


tanh_grad = tanh_derivative(z)
relu_grad = relu_derivative(z)


plt.figure(figsize=(12, 6))

# Plot Relu and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()


# Plot tanH and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, tanh(z), label='tanH Activation', color='g')
plt.plot(z, tanh_grad, label="tanH Derivative", color='r', linestyle='--')
plt.title('tanH Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()