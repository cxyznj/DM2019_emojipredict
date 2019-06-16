from torch import nn, optim, cat
import torch
import numpy as np
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, weight):
        # 继承父类的初始化函数
        super(CNN, self).__init__()
        # 定义卷积层，1 input image channel, 25 output channels, 3*3 square convolution kernel
        # 因为MNIST数据集是黑白图像，所以input是1个channel的，即1个二维张量
        # 输入数据规模为32*32*1，其中最后一维是二维矩阵的数量，前两维是二维矩阵（方阵）的尺寸
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(2, 512)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(31, 1), stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 512)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # ((32-5+2*0)/1)+1 = 28, 28*28*32，其中最后一维由卷积核的数量决定
        # 定义池化层，max pooling over a (2, 2) window
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(30, 1), stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4, 512)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # ((32-5+2*0)/1)+1 = 28, 28*28*32，其中最后一维由卷积核的数量决定
        # 定义池化层，max pooling over a (2, 2) window
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(29, 1), stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(5, 512)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(28, 1), stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(256, 72)
            #nn.ReLU(inplace=True),
            #nn.Linear(256, 72)
        )
        #self.softmax = nn.Softmax()
        weight = weight.type(torch.float32)
        self.embedded = nn.Embedding.from_pretrained(weight, freeze=False)
        self.available_gpu = torch.cuda.is_available()
        #self.available_gpu = False

    def forward(self, x):
        #print(x)
        feature = self.embedded(x)
        feature = torch.unsqueeze(feature, 1)
        '''
        feature = feature.type(torch.float32)
        if self.available_gpu:
            feature = feature.cuda()
        else:
            feature = Variable(feature)
        '''
        #feature = feature.numpy()
        #feature = feature[:, np.newaxis]
        #feature = from_numpy(feature)
        #print(feature)
        f2 = self.conv1(feature)
        #print("x type is", type(f3), ", size is", f3.shape)
        f2 = self.pool1(f2)
        f3 = self.conv2(feature)
        f3 = self.pool2(f3)
        f4 = self.conv3(feature)
        f4 = self.pool3(f4)
        f5 = self.conv4(feature)
        f5 = self.pool4(f5)
        fout = cat((f2, f3, f4, f5), 1)
        fout = fout.view(-1, self.num_flat_features(fout))
        fout = self.fc(fout)
        return fout
        #print("x type is", type(fout), ", size is", fout.shape)
        #x = self.layer3(x)
        #x = self.layer4(x)
        # 官网上给出的展开方式，会增加一定的时间开销
        #x = x.view(-1, self.num_flat_features(x))
        #x = self.fc(x)
        #return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features