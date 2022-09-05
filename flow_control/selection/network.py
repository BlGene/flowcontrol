import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.conv1 = nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(256, 512, stride=1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512 * 7 * 7, 2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.flatten(out)
        out = self.linear(out)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, stride=2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, stride=2, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, stride=2, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mp(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.mp(out)
        out = F.relu(self.bn3(self.conv3(out)))

        return out

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.encoder = Encoder()
        self.neck = Neck()

    def forward(self, x, y):
        out1 = self.encoder(x)
        out2 = self.encoder(y)

        # Stack these outputs along channel dimension
        out = torch.cat((out1, out2), dim=1)

        out = self.neck(out)

        return out

if __name__ == '__main__':
    model = EncoderNet().cuda()
    summary(model, [(3, 256, 256), (3, 256, 256)])
