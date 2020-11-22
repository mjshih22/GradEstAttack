#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from utils import *
from purchase import purchase
import torchvision.transforms as transforms
from estgrad import estgrad

# load pretrained model
target = purchase(num_classes=100)
checkpoint = torch.load('purchase_natural')
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = k
    else:
        k = k.replace('module.', '')
    new_state_dict[k] = v

target.load_state_dict(new_state_dict)
target.eval()

# set criterion
criterion = nn.CrossEntropyLoss()

train_size = 9866
split_size = int(train_size/2)
val_size = split_size
test_size = split_size

#load in data
att_train_train, att_train_test, target_train, target_test = prepare_purchase_data(100)
torch.manual_seed(42)
att_val_train, att_test_train = torch.utils.data.random_split(target_train, [split_size, split_size])
torch.manual_seed(torch.initial_seed())
att_val_test, att_test_test = torch.utils.data.random_split(target_test, [split_size, split_size])

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=50, shuffle=True, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=50, shuffle=True, num_workers=2)
att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=50, shuffle=False, num_workers=2)
att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=50, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=50, shuffle=True, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=50, shuffle=True, num_workers=2)

print("Loading Train Data")
att_train_dataloader = estgrad(target, criterion, att_train_train_loader, att_train_test_loader, train_size, 512, './att_train.t', False)
print("Loading Val Data")
att_val_dataloader = estgrad(target, criterion, att_val_train_loader, att_val_test_loader, val_size, 256, './att_val.t', False)
print("Loading Test Data")
att_test_dataloader = estgrad(target, criterion, att_test_train_loader, att_test_test_loader, test_size, 256, './att_test', False)

print("Data has been loaded")

attack = purchase(num_classes = 2)

# set optimizer
optimizer = optim.Adam(attack.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

best_acc = 0
PATH = './attack_net.pth'

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
        if i % 38 == 37:    # print once per epoch
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 38))
            running_loss = 0.0

	    # evaluate attack mode
            attack.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for data in att_val_dataloader:
                    images, labels = data
                    outputs = attack(images.float())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))

            acc=100*correct/total
            if acc > best_acc:
                print(best_acc)
                best_acc = acc
                torch.save(attack.state_dict(), PATH)

print('Finished Training')

# load best model
attack.load_state_dict(torch.load(PATH))

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
    for data in att_val_dataloader:
        images, labels = data
        outputs = attack(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the validation images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total)) 

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