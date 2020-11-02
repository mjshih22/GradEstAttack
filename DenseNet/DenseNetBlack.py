#!/usr/bin/env python
# coding: utf-8



import torch
import torchvision
import torch.nn as nn
import math
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from membership_inference_attacks import black_box_benchmarks

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

def prepare_model_performance(shadow_model, shadow_train_loader, shadow_test_loader,
                              target_model, target_train_loader, target_test_loader):
    def _model_predictions(model, dataloader):
        return_outputs, return_labels = [], []

        for (inputs, labels) in dataloader:
            return_labels.append(labels.numpy())
            outputs = model.forward(inputs.cuda()) 
            return_outputs.append( softmax_by_row(outputs.data.cpu().numpy()) )
        return_outputs = np.concatenate(return_outputs)
        return_labels = np.concatenate(return_labels)
        return (return_outputs, return_labels)
    
    shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
    shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)
    
    target_train_performance = _model_predictions(target_model, target_train_loader)
    target_test_performance = _model_predictions(target_model, target_test_loader)
    return shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=190, block=Bottleneck, 
        dropRate=0, num_classes=100, growthRate=40, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x)) 
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

densenet = DenseNet()
densenet = densenet.cuda()

PATH = './model_best.pth.tar'

checkpoint = torch.load(PATH)
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = k
    else:
        k = k.replace('module.', '')
    new_state_dict[k] = v

densenet.load_state_dict(new_state_dict)

densenet.eval()

# get 10K from both train and test data
att_train_loader = torch.utils.data.DataLoader(trainset, batch_size=10000,
                                          shuffle=True, num_workers=2)
att_train_enum = enumerate(att_train_loader)
batch_idx, (train_data, train_targets) = next(att_train_enum)


att_test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=True, num_workers=2)
att_test_enum = enumerate(att_test_loader)
batch_idx, (test_data, test_targets) = next(att_test_enum)

# take first 8K and use as attack train data
att_train_train_data = train_data[0:8000]
att_train_train_target = train_targets[0:8000]
att_train_test_data = test_data[0:8000]
att_train_test_target = test_targets[0:8000]

# set up test data
att_test_train_data = train_data[8000:10000]
att_test_train_target = train_targets[8000:10000]
att_test_test_data = test_data[8000:10000]
att_test_test_target = test_targets[8000:10000]


# create data loaders
shadow_train_dataset = torch.utils.data.TensorDataset(att_train_train_data, att_train_train_target)
shadow_train_dataloader = torch.utils.data.DataLoader(shadow_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

shadow_test_dataset = torch.utils.data.TensorDataset(att_train_test_data, att_train_test_target)
shadow_test_dataloader = torch.utils.data.DataLoader(shadow_test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

target_train_dataset = torch.utils.data.TensorDataset(att_test_train_data, att_test_train_target)
target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

target_test_dataset = torch.utils.data.TensorDataset(att_test_test_data, att_test_test_target)
target_test_dataloader = torch.utils.data.DataLoader(target_test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(densenet, shadow_train_dataloader, shadow_test_dataloader, 
                              densenet, target_train_dataloader, target_test_dataloader)

print('Perform membership inference attacks!!!')
MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
            target_train_performance,target_test_performance,num_classes=100)
MIA._mem_inf_benchmarks()




