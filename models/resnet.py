# This implementation is based on the DenseNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, num_input_features, num_output_features, initial_stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=3,
            stride=initial_stride,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(
            num_output_features,
            num_output_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(num_output_features)

        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.initial_stride = initial_stride

    def forward(self, input):
        residual = self.conv1(input)
        residual = self.norm1(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv2(residual)
        residual = self.norm2(residual)

        if self.initial_stride > 1:
            input = F.avg_pool2d(input, 2)
            extras = torch.autograd.Variable(input.data.new(
                input.size(0),
                self.num_output_features - self.num_input_features,
                input.size(2),
                input.size(3),
            ).zero_())
            input = torch.cat([input, extras], 1)

        output = input + residual
        output = F.relu(output)
        return output


class PoolingRegion(nn.Sequential):
    def __init__(self, num_blocks, num_input_features, num_output_features, initial_stride=1):
        layers = [('block%d' % (i + 1), BasicBlock(
            num_input_features=(num_input_features if i == 0 else num_output_features),
            num_output_features=num_output_features,
            initial_stride=(initial_stride if i == 0 else 1),
        )) for i in range(num_blocks)]
        super(PoolingRegion, self).__init__(OrderedDict(layers))


class ResNet(nn.Module):
    '''
    Small ResNet for CIFAR & SVHN
    '''
    def __init__(self, depth, num_features=(16, 32, 64), num_classes=10):
        if (depth - 2) % 6:
            raise RuntimeError('depth should be 6N+2')

        super(ResNet, self).__init__()
        num_blocks = (depth - 2) // 6
        self.avgpool_size = 8

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, num_features[0], kernel_size=3, stride=1, padding=1)),
            ('bn1', nn.BatchNorm2d(num_features[0])),
            ('relu', nn.ReLU(inplace=True)),
            ('layer1', PoolingRegion(num_blocks, num_features[0], num_features[0])),
            ('layer2', PoolingRegion(num_blocks, num_features[0], num_features[1], initial_stride=2)),
            ('layer3', PoolingRegion(num_blocks, num_features[1], num_features[2], initial_stride=2)),
        ]))
        self.classifier = nn.Linear(num_features[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)  # I think this should be a no-op?
        x = F.avg_pool2d(x, kernel_size=self.avgpool_size)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
