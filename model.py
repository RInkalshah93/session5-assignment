import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def _train_transforms(): 
    """_summary_
    """
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),#applying random tranforamtion to given image in this case it is applying center crop with the probeblity of 0.1
        transforms.Resize((28, 28)),#to resize the input image in this case we are changing the input image to 28x28
        transforms.RandomRotation((-15., 15.), fill=0),#this is for randomly rotate the image between -15 to 15 degree and no padding added
        transforms.ToTensor(),#this is to convert any PIL image to tensor formate
        transforms.Normalize((0.1307,), (0.3081,)),#this to normalize the Tensor data for easy data  training
        ])
    return train_transforms


# Test data transformations
def _test_transforms():
    """_summary_
    """
    test_transforms = transforms.Compose([
        transforms.ToTensor(),#this is to convert any PIL image to tensor formate
        transforms.Normalize((0.1307,), (0.3081,))#this to normalize the Tensor data for easy data  testing.
        ])
    return test_transforms
def train_data():
    train_data = datasets.MNIST('../data', train=True, download=True, transform=_train_transforms())#for downloading and training of MNIST data set from PyTorch Torchvision
    return train_data
def test_data():
    test_data = datasets.MNIST('../data', train=False, download=True, transform=_test_transforms())#for downloading and testing of MNIST data set from PyTorch Torchvision
    return test_data

def get_train_loader(batch_size,shuffle,num_workers,pin_memory,train_data):
    """_summary_

    Args:
        batch_size (_type_): _description_
        train_data (_type_): _description_
        shuffle (_type_): _description_
        num_workers (_type_): _description_
        pin_memory (_type_): _description_

    Returns:
        _type_: _description_
    """

    kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'pin_memory': pin_memory}# created dictionary where shuffle of image is allow and specified subprocess to be done using num_workers and to improve the perfomance we used pin memory to tranform the data from cpu to gpu
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)#  to conver dictionary to keyword argument
    return train_loader

def get_test_loader(batch_size,shuffle,num_workers,pin_memory,test_data):
    """_summary_

    Args:
        test_data (_type_): _description_
        batch_size (_type_): _description_
        shuffle (_type_): _description_
        num_workers (_type_): _description_
        pin_memory (_type_): _description_
    """
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'pin_memory': pin_memory}# created dictionary where shuffle of image is allow and specified subprocess to be done using num_workers and to improve the perfomance we used pin memory to tranform the data from cpu to gpu
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)#  to conver dictionary to keyword argument
    return test_loader



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)