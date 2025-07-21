import torch.nn as nn
import torch
import matplotlib.pyplot as plt
torch.manual_seed(2)

z = torch.arange(-10, 10, 0.1,).view(-1, 1)

sig = nn.Sigmoid()

yhat = sig(z)


plt.plot(z.detach().numpy(),yhat.detach().numpy())
plt.xlabel('z')
plt.ylabel('yhat')


yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())

plt.show()


TANH = nn.Tanh()


yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())

yhat = torch.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()



x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.legend()


x = torch.arange(-1, 1, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label = 'relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label = 'sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label = 'tanh')
plt.legend()

plt.show()