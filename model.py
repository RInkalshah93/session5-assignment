import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),#applying random tranforamtion to given image in this case it is applying center crop with the probeblity of 0.1
    transforms.Resize((28, 28)),#to resize the input image in this case we are changing the input image to 28x28
    transforms.RandomRotation((-15., 15.), fill=0),#this is for randomly rotate the image between -15 to 15 degree and no padding added
    transforms.ToTensor(),#this is to convert any PIL image to tensor formate
    transforms.Normalize((0.1307,), (0.3081,)),#this to normalize the Tensor data for easy data  training
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),#this is to convert any PIL image to tensor formate
    transforms.Normalize((0.1307,), (0.3081,))#this to normalize the Tensor data for easy data  testing.
    ])

batch_size = 512 # defined Batch size for processing image in one go

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}# created dictionary where shuffle of image is allow and specified subprocess to be done using num_workers and to improve the perfomance we used pin memory to tranform the data from cpu to gpu

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)#  to conver dictionary to keyword argument
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)#  to conver dictionary to keyword argument

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