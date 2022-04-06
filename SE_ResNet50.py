import torch 
import torch.nn as nn

class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
class ResBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size= 1, stride=stride, padding= 0, bias= False)
        self.bn1= nn.BatchNorm2d(outchannels)
        
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size = 3, stride=1, padding= 1, bias = False)
        self.bn2 = nn.BatchNorm2d(outchannels)

        self.conv3 = nn.Conv2d(outchannels, outchannels*4, kernel_size = 1, stride=1, padding= 0, bias = False)
        self.bn3 = nn.BatchNorm2d(outchannels*4)

        self.se = SELayer(outchannels*4)
        self.relu = nn.ReLU()

        if stride!=1 or inchannels != outchannels*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, outchannels*4, kernel_size= 1, stride= stride, padding= 0, bias= False),
                nn.BatchNorm2d(outchannels*4)
            )
        else: 
            self.shortcut = nn.Sequential()
    
    def forward(self, input_layer):
        shortcut = input_layer 
        x = self.conv1(input_layer)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.se(x)
        x = x + self.shortcut(shortcut)
        x = self.relu(x)
        return x

class SEResNet50(nn.Module):
    def __init__(self):
        super(SEResNet50, self).__init__()

        self.conv_adjust = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride= 2, padding=0, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.resnet = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64*4, 64, 1),
            ResBlock(64*4, 64, 1),
            ResBlock(64*4, 128, 2),
            ResBlock(128*4, 128, 1),
            ResBlock(128*4, 128, 1),
            ResBlock(128*4, 128, 1),
            ResBlock(128*4, 256, 2),
            ResBlock(256*4, 256, 1),
            ResBlock(256*4, 256, 1),
            ResBlock(256*4, 256, 1),
            ResBlock(256*4, 256, 1),
            ResBlock(256*4, 256, 1),
            ResBlock(256*4, 512, 2),
            ResBlock(512*4, 512, 1),
            ResBlock(512*4, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512*4, 168)
        self.fc2 = nn.Linear(512*4, 11)
        self.fc3 = nn.Linear(512*4, 6)
    def forward(self, x):
        x = self.conv_adjust(x)
        x = self.resnet(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3

from torchsummary import summary 
model = SEResNet50()

summary(model, (1, 128, 128))
