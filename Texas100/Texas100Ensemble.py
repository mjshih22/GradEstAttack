#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from ens_utils import *
from texas import texas
from estgrad import estgrad
import torchvision.transforms as transforms

def _entr_comp(prediction):
    entr = 0
    for num in prediction:
        if num != 0:
            entr += -1*num*np.log(num)
    return entr
    
def _m_entr_comp(prediction, label):
    entr = 0
    for i in range(len(prediction)):
        p = prediction[i]
        if i==label:
            if p==0:
                entr += -1*1*np.log(1e-30)
            else:
                entr += -1*(1-p)*np.log(p)
        else:
            if p==1:
                entr += -1*1*np.log(1e-30)
            else:
                entr += -1*p*np.log(1-p)
    return entr
    
def _thre_setting(tr_values, te_values):
    value_list = np.concatenate((tr_values, te_values))
    thre, max_acc = 0, 0
    for value in value_list:
        tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
        te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
        acc = 0.5*(tr_ratio + te_ratio)
        if acc > max_acc:
            thre, max_acc = value, acc
    return thre

def _mem_inf_thre(num_classes, s_tr_values, s_te_values, t_tr_values, t_te_values, s_tr_labels,
                   s_te_labels, t_tr_labels, t_te_labels):
    # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
    # (negative) prediction entropy, and (negative) modified entropy
    predicted = torch.ones(5000)
    thresholds = np.zeros(100)
    for num in range(num_classes):
        thre = _thre_setting(s_tr_values[s_tr_labels==num], s_te_values[s_te_labels==num])
        thresholds[num] = thre

    for i in range(0,2500):
        if(t_tr_values[i]>=thresholds[t_tr_labels[i]]):
                predicted[i] = 0
    for i in range(0,2500):
        if(t_te_values[i]>=thresholds[t_te_labels[i]]):
            predicted[2500+i] = 0
    return predicted

# load pretrained model
target = texas(num_classes=100)
checkpoint = torch.load('texas_advreg')
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
target = target.cuda()
target.eval()

# set criterion
criterion = nn.CrossEntropyLoss()

train_size = 5000
split_size = int(train_size/2)
val_size = split_size
test_size = split_size

#load in data
att_train_train, att_train_test, target_train, target_test = prepare_texas_data(100)
torch.manual_seed(42)
att_val_train, att_test_train = torch.utils.data.random_split(target_train, [split_size, split_size])
torch.manual_seed(torch.initial_seed())
att_val_test, att_test_test = torch.utils.data.random_split(target_test, [split_size, split_size])

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=50, shuffle=True, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=50, shuffle=True, num_workers=2)
att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=50, shuffle=False, num_workers=2)
att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=50, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=test_size, shuffle=True, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=test_size, shuffle=True, num_workers=2)

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

print("Loading Test Data")
att_test_dataloader = estgrad(target, criterion, att_test_train_loader, att_test_test_loader, test_size, 256, './advreg_test.t', True, False)
print("Data has been loaded")

attack = texas(num_classes = 2)

# load best model
PATH = './adv_attack_net3.pth'
attack.load_state_dict(torch.load(PATH))

attack.eval()

# get prediction of gradient
correct = 0
total = 0
predicted_grad = torch.zeros(0, dtype = torch.int64)
with torch.no_grad():
    for data in att_test_dataloader:
        images, labels = data
        outputs = attack(images.float())
        _, pred = torch.max(outputs.data, 1)
        predicted_grad = torch.cat((predicted_grad, pred))
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
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

s_tr_entr = np.array([_entr_comp(s_tr_outputs[i]) for i in range(len(s_tr_labels))])
s_te_entr = np.array([_entr_comp(s_te_outputs[i]) for i in range(len(s_te_labels))])
t_tr_entr = np.array([_entr_comp(t_tr_outputs[i]) for i in range(len(t_tr_labels))])
t_te_entr = np.array([_entr_comp(t_te_outputs[i]) for i in range(len(t_te_labels))])

s_tr_m_entr = np.array([_m_entr_comp(s_tr_outputs[i], s_tr_labels[i]) for i in range(len(s_tr_labels))])
s_te_m_entr = np.array([_m_entr_comp(s_te_outputs[i], s_te_labels[i]) for i in range(len(s_te_labels))])
t_tr_m_entr = np.array([_m_entr_comp(t_tr_outputs[i], t_tr_labels[i]) for i in range(len(t_tr_labels))])
t_te_m_entr = np.array([_m_entr_comp(t_te_outputs[i], t_te_labels[i]) for i in range(len(t_te_labels))])

# get prediction of confidence
predicted_conf = _mem_inf_thre(num_classes, s_tr_conf, s_te_conf, t_tr_conf, t_te_conf, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,test_size):
    if(predicted_conf[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using confidence: {acc:.3f}'.format(acc=100*correct/total))

# get prediction of entropy
predicted_entr = _mem_inf_thre(num_classes, -s_tr_entr, -s_te_entr, -t_tr_entr, -t_te_entr, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,test_size):
    if(predicted_entr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using entropy: {acc:.3f}'.format(acc=100*correct/total))

# get prediction of modified entropy
predicted_m_entr = _mem_inf_thre(num_classes, -s_tr_m_entr, -s_te_m_entr, -t_tr_m_entr, -t_te_m_entr, s_tr_labels,
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