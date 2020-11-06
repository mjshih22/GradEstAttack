#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils import *
from alexnet import alexnet
from membership_inference_attacks import black_box_benchmarks

# import CIFAR100 data
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

# convert data into shadow and target dataloaders. use manual seed to fix dataloaders between instances
train_size = 6000
val_size = 2000
test_size = 2000
data_train_size = 50000
remain_size = data_train_size-train_size-val_size-test_size

torch.manual_seed(42)
att_train_train, att_val_train, att_test_train, _ = torch.utils.data.random_split(trainset, [train_size, val_size ,test_size, remain_size])
torch.manual_seed(torch.initial_seed())
att_train_test, att_val_test, att_test_test = torch.utils.data.random_split(testset, [train_size, val_size, test_size])
torch.manual_seed(torch.initial_seed())


print(len(att_train_train))
print(len(att_train_test))
print(len(att_test_train))
print(len(att_test_test))

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=100, shuffle=True, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=100, shuffle=True, num_workers=2)
# att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=100, shuffle=False, num_workers=2)
# att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=100, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=100, shuffle=True, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=100, shuffle=True, num_workers=2)

print(len(att_train_train_loader.dataset))
print(len(att_train_test_loader.dataset))
print(len(att_test_train_loader.dataset))
print(len(att_test_train_loader.dataset))


# create AlexNet model with pretrained parameters
alexnet = alexnet(num_classes = 100)

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

# perform MIA attacks
shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(alexnet, att_train_train_loader, att_train_test_loader, 
                              alexnet, att_test_train_loader, att_test_train_loader)

test1, _ = shadow_train_performance
test2, _ = shadow_test_performance
test3, _ = target_train_performance
test4, _ = target_test_performance

print(test1.shape)
print(np.linalg.norm(test1))
print(test2.shape)
print(np.linalg.norm(test2))
print(test3.shape)
print(np.linalg.norm(test3))
print(test4.shape)
print(np.linalg.norm(test4))

print('Perform membership inference attacks!!!')
MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
            target_train_performance,target_test_performance,num_classes=100)
MIA._mem_inf_benchmarks()




