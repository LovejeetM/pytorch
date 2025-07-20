import torch.nn as nn
import torch
import matplotlib.pyplot as plt 

torch.manual_seed(2)

z = torch.arange(-100, 100, 0.1).view(-1, 1)
print("The tensor: ", z)

sig = nn.Sigmoid()

yhat = sig(z)

plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)



model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

yhat = model(x)
print("The prediction: ", yhat)

yhat = model(X)
yhat

x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
print('x = ', x)
print('X = ', X)

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

yhat = model(x)
print("The prediction: ", yhat)

yhat = model(X)
print("The prediction: ", yhat)


class logistic_regression(nn.Module):
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat

x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
print('x = ', x)
print('X = ', X)

model = logistic_regression(1)

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

yhat = model(x)
print("The prediction result: \n", yhat)

yhat = model(X)
print("The prediction result: \n", yhat)

model = logistic_regression(2)

x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
print('x = ', x)
print('X = ', X)

yhat = model(x)
print("The prediction result: \n", yhat)

yhat = model(X)
print("The prediction result: \n", yhat)
