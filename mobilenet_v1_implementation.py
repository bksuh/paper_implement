import torch.nn as nn
import torch

class DepthSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=2)
        self.sep_conv2 = DepthSeparableConv2d(32, 64, 1)
        self.sep_conv3 = DepthSeparableConv2d(64, 128, 2)
        self.sep_conv4 = DepthSeparableConv2d(128, 128, 1)
        self.sep_conv5 = DepthSeparableConv2d(128, 256, 2)
        self.sep_conv6 = DepthSeparableConv2d(256, 256, 1)
        self.sep_conv7 = DepthSeparableConv2d(256, 512, 2)
        self.sep_conv8 = DepthSeparableConv2d(512, 512, 1)
        self.sep_conv9 = DepthSeparableConv2d(512, 512, 1)
        self.sep_conv10 = DepthSeparableConv2d(512, 512, 1)
        self.sep_conv11 = DepthSeparableConv2d(512, 512, 1)
        self.sep_conv12 = DepthSeparableConv2d(512, 512, 1)
        self.sep_conv13 = DepthSeparableConv2d(512, 1024, 2)
        self.sep_conv14 = DepthSeparableConv2d(1024, 1024, 2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = self.sep_conv4(x)
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        x = self.sep_conv7(x)
        x = self.sep_conv8(x)
        x = self.sep_conv9(x)
        x = self.sep_conv10(x)
        x = self.sep_conv11(x)
        x = self.sep_conv12(x)
        x = self.sep_conv13(x)
        x = self.sep_conv14(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
