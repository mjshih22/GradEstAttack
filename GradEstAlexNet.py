#!/usr/bin/env python
# coding: utf-8



import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
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
        k = k.replace('features.module.', 'features.')
    new_state_dict[k] = v

alexnet.load_state_dict(new_state_dict)



alexnet.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = alexnet(images)
_, predicted = torch.max(outputs, 1)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = alexnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# set criterion
criterion = nn.CrossEntropyLoss()

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

att_train_data = torch.cat((att_train_train_data, att_train_test_data))
att_train_target = torch.cat((att_train_train_target, att_train_test_target))

# get grad train data
att_train_input = torch.zeros(16000,3,32,32)

for i in range(0, 16000):
    alexnet.zero_grad()
    input = att_train_data[[i]]
    input.requires_grad_(True)
    output = alexnet(input)
    target = att_train_target[[i]]

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    
    att_train_input[i] = g

# make targets 
att_train_label = torch.ones(16000, dtype = torch.int64)
for i in range(0,8000):
    att_train_label[i] = 0

# create data loader
att_train_dataset = torch.utils.data.TensorDataset(att_train_input, att_train_label)
att_train_dataloader = torch.utils.data.DataLoader(att_train_dataset, batch_size=256,
                                          shuffle=True, num_workers=2)

# set up test data
att_test_train_data = train_data[8000:10000]
att_test_train_target = train_targets[8000:10000]
att_test_test_data = test_data[8000:10000]
att_test_test_target = test_targets[8000:10000]

att_test_data = torch.cat((att_test_train_data, att_test_test_data))
att_test_target = torch.cat((att_test_train_target, att_test_test_target))


att_test_input = torch.zeros(4000,3,32,32)

for i in range(0, 4000):
    alexnet.zero_grad()
    input = att_test_data[[i]]
    input.requires_grad_(True)
    output = alexnet(input)
    target = att_test_target[[i]]

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    
    att_test_input[i] = g

att_test_label = torch.ones(4000, dtype = torch.int64)
for i in range(0,2000):
    att_test_label[i] = 0

att_test_dataset = torch.utils.data.TensorDataset(att_test_input, att_test_label)
att_test_dataloader = torch.utils.data.DataLoader(att_test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)


# create attack model
class Attack(nn.Module):

    def __init__(self, num_classes=2):
        super(Attack, self).__init__()
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
    
attack = Attack()

# set optimizer
import torch.optim as optim

optimizer = optim.Adam(attack.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)


# train attack model
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(att_train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = attack(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 60 == 59:    # print once per epoch
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
        	    outputs = attack(images)
        	    _, predicted = torch.max(outputs.data, 1)
        	    total += labels.size(0)
        	    correct += (predicted == labels).sum().item()
	    attack.train

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

print('Finished Training')

PATH = './attack_net.pth'
torch.save(attack.state_dict(), PATH)

# evaluate attack mode
attack.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in att_test_dataloader:
        images, labels = data
        outputs = attack(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
