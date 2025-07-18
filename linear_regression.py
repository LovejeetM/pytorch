import torch
from torch.nn import Linear
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

def forward(x):
    yhat = w * x + b
    return yhat

x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction: ", yhat)

x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)

yhat = forward(x)
print("The prediction: ", yhat)

x = torch.tensor([[1.0], [2.0], [3.0]])

torch.manual_seed(1)

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())

print("weight:",lr.weight)
print("bias:",lr.bias)

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)




x = torch.tensor([[1.0],[2.0],[3.0]])

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)





class plot_diagram():
    def __init__(self, X, Y, w, stop, go = False):
        start = w.data
        self.error = []
        self.parameter = []
        print(type(X.numpy()))
        self.X = X.numpy()
       
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values] 
        w.data = start
        
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y,'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))

        parameter_values_tensor = torch.tensor(self.parameter_values)
        loss_function_tensor = torch.tensor(self.Loss_function)

        plt.plot(parameter_values_tensor.numpy(), loss_function_tensor.numpy())
  
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
    

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X


plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

Y = f + 0.1 * torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')

plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

def forward(x):
    return w * x

def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

lr = 0.1
LOSS = []

w = torch.tensor(-10.0, requires_grad = True)
gradient_plot = plot_diagram(X, Y, w, stop = 5)




def train_model(iter):
    for epoch in range (iter):
        Yhat = forward(X)
        loss = criterion(Yhat,Y)

        gradient_plot(Yhat, w, loss.item(), epoch)
        LOSS.append(loss.item())
        loss.backward()
        
        w.data = w.data - lr * w.grad.data
        
        w.grad.data.zero_()

train_model(4)

plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.show()
