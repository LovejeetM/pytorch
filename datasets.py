import torch
from torch.utils.data import Dataset
from torchvision import transforms
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