import torch
import torch.utils.data.dataloader as dataloader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from sklearn import metrics
from Training import Training
from Visualization import *

class MNIST_Model(nn.Module,Training):
    
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.max_pool2d(x1, 2, 2)
        x = F.dropout(x,0.2)

        x2 = F.relu(self.conv2(x))
        x = F.max_pool2d(x2, 2, 2)
        x = F.dropout(x,0.2)

        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        h = F.softmax(x,dim = 1)

        return h, x1, x2

class FashionMNIST_Model(nn.Module,Training):
    
    def __init__(self):
        super(FashionMNIST_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1= nn.Conv2d(1,64,5)
        self.conv2= nn.Conv2d(64,32,5)
        self.conv3= nn.Conv2d(32,16,3)
        
        self.fc1 = nn.Linear(16*1*1, 128)
        self.fc2 = nn.Linear(128,32)
        self.out = nn.Linear(32,10)
       
    def forward(self,x):
        #1st block
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x = F.max_pool2d(x1,2,2)
        
        #2nd block
        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x = F.max_pool2d(x2,2,2)
        x = F.dropout(x,0.5)
        
        #3rd block
        x3 = self.conv3(x)
        x3 = F.relu(x3)
        x = F.max_pool2d(x3,2,2)
        
        #1st FC layer
        x = x.view(-1, 16*1*1) #reshape
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x,0.2)
        
        #2nd FC layer
        x = F.relu(self.fc2(x))
        
        #Output Layer
        h = self.out(x)
        h = F.softmax(h,dim=1)
        
        return h, x1, x2, x3
    
class AlexNet(nn.Module,Training):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 9216, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=10)


    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x = F.max_pool2d(x1,3,2)
        x2 = F.relu(self.conv2(x))
        x = F.max_pool2d(x2,3,2)
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x = F.max_pool2d(x5,3,2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, x1,x2,x3,x4,x5
    