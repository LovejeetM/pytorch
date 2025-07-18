import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from PIL import Image
import pandas as pd
import os

torch.manual_seed(1)


class toy_set(Dataset):
    
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    def __len__(self):
        return self.len
    

our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))

for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)


class add_mult(object):
    
    def __init__(self, addx = 1, muly = 2):
        self.addx = addx
        self.muly = muly
    
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample
    
a_m = add_mult()
data_set = toy_set()

for i in range(10):
    x, y = data_set[i]
    print('I: ', i, 'x: ', x, 'y: ', y)
    x_, y_ = a_m(data_set[i])
    print('I: ', i, 'Transformed x:', x_, 'Transformed y:', y_)


class mult(object):
    def __init__(self, mult = 100):
        self.mult = mult
        
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample

data_transform = transforms.Compose([add_mult(), mult()])
print("transforms: ", data_transform)


data_transform(data_set[0])

x,y=data_set[0]
x_,y_=data_transform(data_set[0])
print( 'x: ', x, 'y: ', y)

print( 'Transformed x:', x_, 'Transformed y:', y_)

compose_data_set = toy_set(transform = data_transform)

cust_data_set = toy_set(transform = a_m)

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'x: ', x, 'y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)






# image datasets -------------------------------------------------------------------------------------

torch.manual_seed(0)

directory=""
csv_file ='index.csv'
csv_path=os.path.join(directory,csv_file)

data_name = pd.read_csv(csv_path)
print(data_name.head())

print('File name:', data_name.iloc[0, 1])
print('y:', data_name.iloc[0, 0])
print('File name:', data_name.iloc[1, 1])
print('class or y:', data_name.iloc[1, 0])
print('The number of rows: ', data_name.shape[0])

image_name =data_name.iloc[1, 1]
print(image_name)

image_path=os.path.join(directory,image_name)
print(image_path)

image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[1, 0])
plt.show()

image_name = data_name.iloc[19, 1]
image_path=os.path.join(directory,image_name)
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[19, 0])
plt.show()

class Dataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir=data_dir
        self.transform = transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        self.data_name= pd.read_csv(data_dircsv_file)
        
        self.len=self.data_name.shape[0] 
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        image = Image.open(img_name)
        
        y = self.data_name.iloc[idx, 0]
        
        if self.transform:
            image = self.transform(image)

        return image, y
    
dataset = Dataset(csv_file=csv_file, data_dir=directory)

image=dataset[0][0]
y=dataset[0][1]

plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()

print(y)

image=dataset[9][0]
y=dataset[9][1]

plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=croptensor_data_transform )
print("The shape of the first element tensor: ", dataset[0][0].shape)




dataset = dsets.MNIST(
    root = './data',  
    download = True, 
    transform = transforms.ToTensor()
)




print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))
print("As the result, the structure of the first element in the dataset is (tensor([1, 28, 28]), tensor(7)).")

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

show_data(dataset[0])

show_data(dataset[1])

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', download = True, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)

show_data(dataset[0],shape = (20, 20))

show_data(dataset[1],shape = (20, 20))

fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', download = True, transform = fliptensor_data_transform)
show_data(dataset[1])
