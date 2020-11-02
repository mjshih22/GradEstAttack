#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import tarfile
from utils import *
import torchvision.transforms as transforms

# define model
class TexasClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(TexasClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

# load pretrained model
model = TexasClassifier(num_classes=100)
checkpoint = torch.load('texas_natural')
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = k
    else:
        k = k.replace('module.', '')
    new_state_dict[k] = v
model.load_state_dict(new_state_dict)

model.eval()

# set criterion
criterion = nn.CrossEntropyLoss()

#load in data
att_train_train_loader, att_train_test_loader, att_test_train_loader, att_test_test_loader = prepare_texas_data(100)

data_size = 10000

# get grad on train data
for i, data in enumerate(att_train_train_loader):
    model.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = model(input.float())

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    
    if i == 0:
        att_train_input = g
    else:
        att_train_input = torch.cat((att_train_input, g))

print(att_train_input.shape)

for i, data in enumerate(att_train_test_loader):
    model.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = model(input.float())

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    att_train_input = torch.cat((att_train_input, g))

print(att_train_input.shape)

# make targets 
att_train_label = torch.ones(data_size, dtype = torch.int64)
for i in range(0,int(data_size/2)):
    att_train_label[i] = 0

print(att_train_label.shape)

# create data loader
att_train_dataset = torch.utils.data.TensorDataset(att_train_input, att_train_label)
att_train_dataloader = torch.utils.data.DataLoader(att_train_dataset, batch_size=256,
                                          shuffle=True, num_workers=2)

# get grad on test data
for i, data in enumerate(att_test_train_loader):
    model.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = model(input.float())

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    
    if i == 0:
        att_test_input = g
    else:
        att_test_input = torch.cat((att_test_input, g))
        
print(att_test_input.shape)

for i, data in enumerate(att_test_test_loader):
    model.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = model(input.float())

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    att_test_input = torch.cat((att_test_input, g))

print(att_test_input.shape)

# make targets 
att_test_label = torch.ones(data_size, dtype = torch.int64)
for i in range(0,int(data_size/2)):
    att_test_label[i] = 0

print(att_test_label.shape)

# create data loader
att_test_dataset = torch.utils.data.TensorDataset(att_test_input, att_test_label)
att_test_dataloader = torch.utils.data.DataLoader(att_test_dataset, batch_size=500,
                                          shuffle=False, num_workers=2)

print("Data has been loaded")

# create attack model
class Attack(nn.Module):
    def __init__(self,num_classes=2):
        super(Attack, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)
    
attack = Attack()

# set optimizer
import torch.optim as optim

optimizer = optim.Adam(attack.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

# train attack model
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(att_train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = attack(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 35 == 34:    # print once per epoch
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 60))
            running_loss = 0.0

	    # evaluate attack mode
            attack.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for data in att_test_dataloader:
                    images, labels = data
                    outputs = attack(images.float())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))

            attack.train()

print('Finished Training')

PATH = './attack_net.pth'
torch.save(attack.state_dict(), PATH)

# evaluate attack mode
attack.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in att_train_dataloader:
        images, labels = data
        outputs = attack(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the train images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))

correct = 0
total = 0
with torch.no_grad():
    for data in att_test_dataloader:
        images, labels = data
        outputs = attack(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))