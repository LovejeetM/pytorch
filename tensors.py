import torch 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

tensor1 = torch.tensor([1,2,3,4,5,5])
print(tensor1)
print(tensor1.dtype)
print(tensor1.type())

print(tensor1.size())
print(tensor1 + 1)
print(tensor1.log10())
print(tensor1.tanh())

print(tensor1.tan().dtype)


b = torch.tensor(tensor1.sin(), dtype = torch.int32)
print(b)

new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()
print("The type of the new_float_tensor:", new_float_tensor.type())

print(tensor1.ndimension())

tensor2 = torch.tensor([[1,4,5,5,5],[4,5,3,6,7]])
print(tensor2.ndimension())

numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)

print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype)

a = tensor2.numpy()
print(a)
c = torch.from_numpy(a)
print(c)