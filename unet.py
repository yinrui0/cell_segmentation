import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class UnetDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UnetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, concat_channel, in_channel, out_channel):
        super(UNetUpBlock, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(concat_channel + in_channel, out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, prev_x, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_x), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UnetDownBlock(3, 32)
        self.down_block2 = UnetDownBlock(32, 64)
        self.down_block3 = UnetDownBlock(64, 128)
        self.down_block4 = UnetDownBlock(128, 256)
        self.down_block5 = UnetDownBlock(256, 512)

        self.up_block1 = UNetUpBlock(256, 512, 256)
        self.up_block2 = UNetUpBlock(128, 256, 128)
        self.up_block3 = UNetUpBlock(64, 128, 64)
        self.up_block4 = UNetUpBlock(32, 64, 32)

        self.last_conv1 = nn.Conv2d(32, 1, 1, padding=0)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        c1 = self.down_block1(x)
        x = self.pool1(c1)
        c2 = self.down_block2(x)
        x = self.pool1(c2)
        c3 = self.down_block3(x)
        x = self.pool1(c3)
        c4 = self.down_block4(x)
        x = self.pool1(c4)
        x = self.down_block5(x)

        x = self.up_block1(c4, x)
        x = self.up_block2(c3, x)
        x = self.up_block3(c2, x)
        x = self.up_block4(c1, x)

        x = self.last_conv1(x)
        #x = F.sigmoid(x)

        return x

if __name__ == '__main__':
    net = UNet().cuda()
    print(net)

    test_x = Variable(torch.FloatTensor(1, 3, 256, 256)).cuda()
    out_x = net(test_x)

    print(out_x.size())