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
    def __init__(self, inchannels, outchannels, kernel_size, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size, stride= stride, padding= 1, bias= False)
        self.bn1= nn.BatchNorm2d(outchannels)
        
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size, stride=1, padding= 1, bias = False)
        self.bn2 = nn.BatchNorm2d(outchannels)

        self.se = SELayer(outchannels)
        self.relu = nn.ReLU()

        if stride!=1 or inchannels != outchannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, kernel_size, stride= stride, padding= 1, bias= False),
                nn.BatchNorm2d(outchannels)
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
        x = self.se(x)
        x = x + self.shortcut(shortcut)
        x = self.relu(x)
        return x
        
class SEResNet34(nn.Module):
    def __init__(self,num_classes=8631):
        super(SEResNet34, self).__init__()
        self.num_classes = num_classes
        self.conv_adjust = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1, bias= False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.resnet = nn.Sequential(
            ResBlock(64, 64,3, 1),
            ResBlock(64, 64, 3, 1),
            ResBlock(64, 64, 3, 1),
            ResBlock(64, 128, 3, 2),
            ResBlock(128, 128, 3, 1),
            ResBlock(128, 128, 3, 1),
            ResBlock(128, 128, 3, 1),
            ResBlock(128, 256, 3, 2),
            ResBlock(256, 256, 3, 1),
            ResBlock(256, 256, 3, 1),
            ResBlock(256, 256, 3, 1),
            ResBlock(256, 256, 3, 1),
            ResBlock(256, 256, 3, 1),
            ResBlock(256, 512, 3, 2),
            ResBlock(512, 512, 3, 1),
            ResBlock(512, 512, 3, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # vowel_diacritic
        self.fc1 = nn.Linear(512,2048)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(2048,self.num_classes)
    def forward(self, x):
        x = self.conv_adjust(x)
        x = self.resnet(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x1 = self.relu_fc1(x)
        x2 = self.fc2(x1)
        return x1,x2
        
from torchsummary import summary 
model = SEResNet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
summary(model, (1, 128, 128))
