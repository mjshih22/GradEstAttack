#!/usr/bin/env python
# coding: utf-8



import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils import *
from membership_inference_attacks import black_box_benchmarks

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

att_train_train, att_test_train, _ = torch.utils.data.random_split(trainset, [8000,2000, 40000], generator=torch.Generator().manual_seed(42))
att_train_test, att_test_test = torch.utils.data.random_split(testset, [8000,2000], generator=torch.Generator().manual_seed(42))

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=100, shuffle=False, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=100, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=100, shuffle=False, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=100, shuffle=False, num_workers=2)

import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
alexnet = AlexNet()


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

alexnet.load_state_dict(new_state_dict)

alexnet.eval()

shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(alexnet, att_train_train_loader, att_train_test_loader, 
                              alexnet, att_test_train_loader, att_test_train_loader)

print('Perform membership inference attacks!!!')
MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
            target_train_performance,target_test_performance,num_classes=100)
MIA._mem_inf_benchmarks()




