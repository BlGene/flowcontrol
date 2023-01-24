import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.encoder = Encoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            # with torch.no_grad():
            x = self.encoder(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(256, 128, stride=1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.flatten(out)
        out = self.linear(out)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, stride=2, kernel_size=3)
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

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1, depth=34):
        super(ResNetEncoder, self).__init__()
        # ipdb.set_trace()

        if depth == 18:
            self.res_encoder = torchvision.models.resnet18(pretrained=False)
        elif depth == 34:
            self.res_encoder = torchvision.models.resnet34(pretrained=False)
        elif depth == 50:
            self.res_encoder = torchvision.models.resnet50(pretrained=False)
        else:
            raise NotImplementedError

        arch = list(self.res_encoder.children())
        w = arch[0].weight

        if in_channels != 3:
            # Modify the first conv layer based on the number of input channels
            arch[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=2, bias=False)
            # arch[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))

        # Delete the last two layers - avg_pool, fc        
        del arch[-2]
        del arch[-1]

        self.res_encoder = nn.Sequential(*arch)
    
    def forward(self, x):
        return self.res_encoder(x)

class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()

        # in_channels = 6 ---> 3 from live image and 3 from demo image
        self.encoder = ResNetEncoder(in_channels=6, depth=18)
        self.neck = Neck()
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)

        return self.sigmoid(x)

if __name__ == '__main__':
    model = SimilarityNet().cuda()
    summary(model, [(6, 256, 256)])
