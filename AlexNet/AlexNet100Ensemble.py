#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import numpy as np
from utils import *
from estgrad import estgrad
from alexnet import alexnet
import torchvision.transforms as transforms
import torch.nn as nn


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
train_size = 8000
val_size = 1000
test_size = 1000
data_train_size = 50000
remain_size = data_train_size-train_size-val_size-test_size

torch.manual_seed(42)
att_train_train, att_val_train, att_test_train, _ = torch.utils.data.random_split(trainset, [train_size, val_size, test_size, remain_size])
torch.manual_seed(torch.initial_seed())
att_train_test, att_val_test, att_test_test = torch.utils.data.random_split(testset, [train_size, val_size, test_size])
torch.manual_seed(torch.initial_seed())

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=256, shuffle=False, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=256, shuffle=False, num_workers=2)
att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=256, shuffle=False, num_workers=2)
att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=256, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=test_size, shuffle=False, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=test_size, shuffle=False, num_workers=2)

# prepare test data
for data in att_test_train_loader:
    att_test_train_data, att_test_train_target = data
for data in att_test_test_loader:
    att_test_test_data, att_test_test_target = data
att_test_data = torch.cat((att_test_train_data, att_test_test_data))
att_test_target = torch.cat((att_test_train_target, att_test_test_target))

# make targets 
att_test_label = torch.ones(2*test_size, dtype = torch.int64)
for i in range(0,test_size):
    att_test_label[i] = 0

# create AlexNet model with pretrained parameters
target = alexnet(num_classes = 100)

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
target.load_state_dict(new_state_dict)

target = nn.DataParallel(target).cuda()
target.eval()

# set criterion for loss
criterion = nn.CrossEntropyLoss()

# create data loader with gradients for attack
print("Loading Test Data")
att_test_dataloader = estgrad(target, criterion, att_test_train_loader, att_test_test_loader, test_size, 256, './att_test_unshuffle.npy', True)
print("Data has been loaded")

# create attack model
attack = alexnet(num_classes = 2)
PATH = './attack_net5.pth'
attack = nn.DataParallel(attack).cuda()
attack.load_state_dict(torch.load(PATH))

# evaluate attack mode
attack.eval()

# get prediction of gradient
correct = 0
total = 0
predicted_grad = torch.zeros(0, dtype = torch.int64).cuda()
with torch.no_grad():
    for data in att_test_dataloader:
        images, labels = data
        outputs = attack(images.float().cuda())
        _, pred = torch.max(outputs.data, 1)
        predicted_grad = torch.cat((predicted_grad, pred))
        total += labels.size(0)
        correct += (pred == labels.cuda()).sum().item()
        
print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))

#get prediction of correctness
test_size = 2*test_size
predicted_corr = torch.ones(test_size)
correct = 0
for i in range(0, test_size):
    target.zero_grad()
    input = att_test_data[[i]]
    output = target(input.float().cuda())
    _, predicted = torch.max(output.data, 1)
    label = att_test_target[[i]]
    if(predicted == label.cuda()):
        predicted_corr[i] = 0

for i in range(0,test_size):
    if(predicted_corr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using correctness: {acc:.3f}'.format(acc=100*correct/total))

# create blackbox_attacks variables
shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(target, att_train_train_loader, att_train_test_loader, 
                              target, att_test_train_loader, att_test_test_loader)

num_classes = 100
        
s_tr_outputs, s_tr_labels = shadow_train_performance
s_te_outputs, s_te_labels = shadow_test_performance
t_tr_outputs, t_tr_labels = target_train_performance
t_te_outputs, t_te_labels = target_test_performance

s_tr_conf = np.array([s_tr_outputs[i, s_tr_labels[i]] for i in range(len(s_tr_labels))])
s_te_conf = np.array([s_te_outputs[i, s_te_labels[i]] for i in range(len(s_te_labels))])
t_tr_conf = np.array([t_tr_outputs[i, t_tr_labels[i]] for i in range(len(t_tr_labels))])
t_te_conf = np.array([t_te_outputs[i, t_te_labels[i]] for i in range(len(t_te_labels))])

s_tr_entr = np.array([entr_comp(s_tr_outputs[i]) for i in range(len(s_tr_labels))])
s_te_entr = np.array([entr_comp(s_te_outputs[i]) for i in range(len(s_te_labels))])
t_tr_entr = np.array([entr_comp(t_tr_outputs[i]) for i in range(len(t_tr_labels))])
t_te_entr = np.array([entr_comp(t_te_outputs[i]) for i in range(len(t_te_labels))])

s_tr_m_entr = np.array([m_entr_comp(s_tr_outputs[i], s_tr_labels[i]) for i in range(len(s_tr_labels))])
s_te_m_entr = np.array([m_entr_comp(s_te_outputs[i], s_te_labels[i]) for i in range(len(s_te_labels))])
t_tr_m_entr = np.array([m_entr_comp(t_tr_outputs[i], t_tr_labels[i]) for i in range(len(t_tr_labels))])
t_te_m_entr = np.array([m_entr_comp(t_te_outputs[i], t_te_labels[i]) for i in range(len(t_te_labels))])

# get prediction of confidence
predicted_conf = mem_inf_thre(num_classes, s_tr_conf, s_te_conf, t_tr_conf, t_te_conf, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,test_size):
    if(predicted_conf[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using confidence: {acc:.3f}'.format(acc=100*correct/total))

# get prediction of entropy
predicted_entr = mem_inf_thre(num_classes, -s_tr_entr, -s_te_entr, -t_tr_entr, -t_te_entr, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,test_size):
    if(predicted_entr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using entropy: {acc:.3f}'.format(acc=100*correct/total))

# get prediction of modified entropy
predicted_m_entr = mem_inf_thre(num_classes, -s_tr_m_entr, -s_te_m_entr, -t_tr_m_entr, -t_te_m_entr, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,test_size):
    if(predicted_m_entr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using modified entropy: {acc:.3f}'.format(acc=100*correct/total))

predicted_ens = torch.zeros(test_size)
for i in range(0,test_size):
    ens = predicted_grad[i] + predicted_corr[i] + predicted_conf[i] + predicted_entr[i] + predicted_m_entr[i]
    if(ens >= 3):
        predicted_ens[i] = 1

correct = 0
for i in range(0,test_size):
    if(predicted_ens[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using ensemble: {acc:.3f}'.format(acc=100*correct/total))
