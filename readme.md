# PyTorch Implementations and Deep Learning Concepts

This repository serves as a comprehensive collection of scripts demonstrating various deep learning models and foundational concepts, implemented using the PyTorch framework. The code is organized to provide clear, focused examples ranging from the mathematical underpinnings of neural networks to the implementation of advanced architectures.

---

## Scripts Overview

The repository is structured into several key areas of study, with each script targeting a specific concept.

### 1. Foundational Concepts

This section covers the core mathematical and programming principles behind neural networks.

* `tensors.py`: Introduction to PyTorch Tensors, the fundamental data structure for all operations.
* `derivative.py`: A practical demonstration of computing gradients with PyTorch's autograd engine.
* `back_propagation.py`: An implementation illustrating the backpropagation algorithm from a conceptual level.
* `activation_functions.py`, `activation_functions1.py`, `activation_function.py`: Exploration and visualization of various activation functions (e.g., ReLU, Sigmoid, Tanh) and their impact.
* `softmax_function.py`, `softmax_function1.py`: Implementation of the Softmax function for multi-class probability distributions.
* `cross_entropy_loss.py`: A script detailing the implementation and use of Cross-Entropy Loss for classification tasks.

### 2. Linear Models

Implementations of classic linear models, forming the basis for more complex neural networks.

* `linear_regression.py`, `linear_regression1.py`: Building a simple linear regression model to predict a single continuous output.
* `multiple_linear_regression.py`, `multiple_linear_regression1.py`: Extending linear regression to handle multiple input features.
* `logistic_regression.py`, `logistic_regression1.py`: Implementation of logistic regression for binary classification problems.

### 3. Neural Network Architectures

Scripts demonstrating the construction of neural networks with increasing complexity.

* `one_layer_nn.py`: The most basic neural network with a single hidden layer.
* `neural_network.py`, `neural_network1.py`: Building a standard feed-forward neural network (Multi-Layer Perceptron).
* `multi_dimensional_nn.py`, `multi_dimensional_nn1.py`: Handling multi-dimensional input and output data with neural networks.
* `deep_nn.py`, `deep_nn1.py`: Constructing deep neural networks with multiple hidden layers.
* `multiple_outputs.py`: Designing models that predict multiple target variables simultaneously.

### 4. Training and Optimization Techniques

This section focuses on the process of training models and methods for improving performance and stability.

* `SGD_model.py`: Demonstrates the use of Stochastic Gradient Descent for model optimization.
* `min_batch_GD.py`: Implementation of mini-batch gradient descent, balancing computational efficiency and gradient stability.
* `optimizer.py`: An exploration of various PyTorch optimizers beyond SGD, such as Adam or RMSprop.
* `validation_data.py`: Illustrates the correct procedure for using a separate validation set to monitor model performance and prevent overfitting.
* `dropout_function.py`, `dropout_function1.py`: Implementing dropout as a regularization technique to reduce overfitting in neural networks.
* `xavier_uniform.py`, `same_weights.py`: Scripts exploring different weight initialization strategies and their effects on training dynamics.

### 5. Advanced Architectures

Implementations of state-of-the-art deep learning models.

* `CNN_problem.py`: Building a Convolutional Neural Network (CNN) for image classification tasks.
* `Transformer.py`: An implementation of the Transformer architecture, focusing on its core self-attention mechanism, which is fundamental to modern NLP.

### 6. Data Handling

Scripts related to loading and managing datasets within PyTorch.

* `datasets.py`: Demonstrates how to create custom `Dataset` classes in PyTorch for loading and preprocessing unique data formats.

---

## Key Concepts Covered

This repository provides practical examples of the following deep learning topics:

* **Core PyTorch API**: Tensors, Autograd, and `nn.Module`.
* **Model Building**: Constructing linear models, feed-forward neural networks, and CNNs.
* **Loss Functions**: MSE Loss, Cross-Entropy Loss.
* **Optimization**: Gradient Descent (Stochastic, Mini-batch) and advanced optimizers (e.g., Adam).
* **Regularization**: Implementation and effects of Dropout.
* **Training Procedures**: Using training/validation splits and monitoring performance.
* **Advanced Architectures**: Implementation of Transformers and CNNs.
* **Data Management**: Creating custom datasets for use with `DataLoader`.