#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import numpy as np
from utils import *
from resnet import resnet
import torch.optim as optim
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

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=train_size, shuffle=False, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=train_size, shuffle=False, num_workers=2)
att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=val_size, shuffle=False, num_workers=2)
att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=val_size, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=test_size, shuffle=False, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=test_size, shuffle=False, num_workers=2)

for data in att_train_train_loader:
    att_train_train_data, att_train_train_target = data
for data in att_train_test_loader:
    att_train_test_data, att_train_test_target = data
train_data = torch.cat((att_train_train_data, att_train_test_data))
train_target = torch.cat((att_train_train_target, att_train_test_target))

for data in att_val_train_loader:
    att_val_train_data, att_val_train_target = data
for data in att_val_test_loader:
    att_val_test_data, att_val_test_target = data
val_data = torch.cat((att_val_train_data, att_val_test_data))
val_target = torch.cat((att_val_train_target, att_val_test_target))

for data in att_test_train_loader:
    att_test_train_data, att_test_train_target = data
for data in att_test_test_loader:
    att_test_test_data, att_test_test_target = data
test_data = torch.cat((att_test_train_data, att_test_test_data))
test_target = torch.cat((att_test_train_target, att_test_test_target))

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=100, shuffle=False, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=100, shuffle=False, num_workers=2)
att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=100, shuffle=False, num_workers=2)
att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=100, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=100, shuffle=False, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=100, shuffle=False, num_workers=2)

# create ResNet model with pretrained parameters
target = resnet(depth = 164, block_name='bottleNeck')
target = nn.DataParallel(target).cuda()

PATH = './model_best.pth.tar'

checkpoint = torch.load(PATH)
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = 'module.' + k
    else:
        k = k
    new_state_dict[k] = v

target.load_state_dict(new_state_dict)
target.eval()

# set criterion for loss
criterion = nn.CrossEntropyLoss()


grad_train_data = np.load('att_train.npy')
grad_train_data = torch.from_numpy(grad_train_data)
grad_val_data = np.load('att_val.npy')
grad_val_data = torch.from_numpy(grad_val_data)
grad_test_data = np.load('att_test.npy')
grad_test_data = torch.from_numpy(grad_test_data)
print("Grad Data has be loaded")

att_train_label = torch.ones(2*train_size, dtype = torch.int64)
for i in range(0,train_size):
    att_train_label[i] = 0
att_val_label = torch.ones(2*val_size, dtype = torch.int64)
for i in range(0,val_size):
    att_val_label[i] = 0
att_test_label = torch.ones(2*test_size, dtype = torch.int64)
for i in range(0,test_size):
    att_test_label[i] = 0
print("Labels have been loaded")

#get correctness, 0 = correct
# also get output and labels
corr_train_data = torch.ones(2*train_size)
labels_train_input = torch.zeros(2*train_size, 100)

for i in range(0, 2*train_size):
    target.zero_grad()
    input = train_data[[i]]
    output = target(input.cuda())
    if i == 0:
        output_train_input = output.data.cpu()
    else:
        output_train_input = torch.cat((output_train_input, output.data.cpu()))
    _, predicted = torch.max(output.data.cpu(), 1)
    label = train_target[[i]]
    labels_train_input[i][label] = 1
    if(predicted == label):
        corr_train_data[i] = 0

corr_val_data = torch.ones(2*val_size)
labels_val_input = torch.zeros(2*val_size, 100)

for i in range(0, 2*val_size):
    target.zero_grad()
    input = val_data[[i]]
    output = target(input.cuda())
    if i == 0:
        output_val_input = output.data.cpu()
    else:
        output_val_input = torch.cat((output_val_input, output.data.cpu()))
    _, predicted = torch.max(output.data.cpu(), 1)
    label = val_target[[i]]
    labels_val_input[i][label] = 1
    if(predicted == label):
        corr_val_data[i] = 0

corr_test_data = torch.ones(2*test_size)
labels_test_input = torch.zeros(2*test_size, 100)

for i in range(0, 2*test_size):
    target.zero_grad()
    input = test_data[[i]]
    output = target(input.cuda())
    if i == 0:
        output_test_input = output.data.cpu()
    else:
        output_test_input = torch.cat((output_test_input, output.data.cpu()))
    _, predicted = torch.max(output.data.cpu(), 1)
    label = test_target[[i]]
    labels_test_input[i][label] = 1
    if(predicted == label):
        corr_test_data[i] = 0

print("Correctness Data Loaded")

# create blackbox_attacks variables for test set
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
conf_train_data = np.concatenate((s_tr_conf, s_te_conf))
conf_train_data = torch.from_numpy(conf_train_data)

t_tr_conf = np.array([t_tr_outputs[i, t_tr_labels[i]] for i in range(len(t_tr_labels))])
t_te_conf = np.array([t_te_outputs[i, t_te_labels[i]] for i in range(len(t_te_labels))])
conf_test_data = np.concatenate((t_tr_conf, t_te_conf))
conf_test_data = torch.from_numpy(conf_test_data)

s_tr_entr = np.array([entr_comp(s_tr_outputs[i]) for i in range(len(s_tr_labels))])
s_te_entr = np.array([entr_comp(s_te_outputs[i]) for i in range(len(s_te_labels))])
entr_train_data = np.concatenate((s_tr_entr, s_te_entr))
entr_train_data = torch.from_numpy(entr_train_data)

t_tr_entr = np.array([entr_comp(t_tr_outputs[i]) for i in range(len(t_tr_labels))])
t_te_entr = np.array([entr_comp(t_te_outputs[i]) for i in range(len(t_te_labels))])
entr_test_data = np.concatenate((t_tr_entr, t_te_entr))
entr_test_data = torch.from_numpy(entr_test_data)

s_tr_m_entr = np.array([m_entr_comp(s_tr_outputs[i], s_tr_labels[i]) for i in range(len(s_tr_labels))])
s_te_m_entr = np.array([m_entr_comp(s_te_outputs[i], s_te_labels[i]) for i in range(len(s_te_labels))])
m_entr_train_data = np.concatenate((s_tr_m_entr, s_te_m_entr))
m_entr_train_data = torch.from_numpy(m_entr_train_data)

t_tr_m_entr = np.array([m_entr_comp(t_tr_outputs[i], t_tr_labels[i]) for i in range(len(t_tr_labels))])
t_te_m_entr = np.array([m_entr_comp(t_te_outputs[i], t_te_labels[i]) for i in range(len(t_te_labels))])
m_entr_test_data = np.concatenate((t_tr_m_entr, t_te_m_entr))
m_entr_test_data = torch.from_numpy(m_entr_test_data)
  
# create blackbox linear input
bb_train_input = torch.zeros(2*train_size, 4)
for i in range(0,2*train_size):
    bb_train_input[i][0] = corr_train_data[i]
    bb_train_input[i][1] = conf_train_data[i]
    bb_train_input[i][2] = entr_train_data[i]
    bb_train_input[i][3] = m_entr_train_data[i]

bb_test_input = torch.zeros(2*test_size, 4)
for i in range(0,2*test_size):
    bb_test_input[i][0] = corr_test_data[i]
    bb_test_input[i][1] = conf_test_data[i]
    bb_test_input[i][2] = entr_test_data[i]
    bb_test_input[i][3] = m_entr_test_data[i]


# create blackbox_attacks variables for validation set
shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(target, att_train_train_loader, att_train_test_loader, 
                              target, att_val_train_loader, att_val_test_loader)

num_classes = 100
        
t_tr_outputs, t_tr_labels = target_train_performance
t_te_outputs, t_te_labels = target_test_performance

t_tr_conf = np.array([t_tr_outputs[i, t_tr_labels[i]] for i in range(len(t_tr_labels))])
t_te_conf = np.array([t_te_outputs[i, t_te_labels[i]] for i in range(len(t_te_labels))])
conf_val_data = np.concatenate((t_tr_conf, t_te_conf))
conf_val_data = torch.from_numpy(conf_val_data)

t_tr_entr = np.array([entr_comp(t_tr_outputs[i]) for i in range(len(t_tr_labels))])
t_te_entr = np.array([entr_comp(t_te_outputs[i]) for i in range(len(t_te_labels))])
entr_val_data = np.concatenate((t_tr_entr, t_te_entr))
entr_val_data = torch.from_numpy(entr_val_data)

t_tr_m_entr = np.array([m_entr_comp(t_tr_outputs[i], t_tr_labels[i]) for i in range(len(t_tr_labels))])
t_te_m_entr = np.array([m_entr_comp(t_te_outputs[i], t_te_labels[i]) for i in range(len(t_te_labels))])
m_entr_val_data = np.concatenate((t_tr_m_entr, t_te_m_entr))
m_entr_val_data = torch.from_numpy(m_entr_val_data)

bb_val_input = torch.zeros(2*val_size, 4)
for i in range(0,2*val_size):
    bb_val_input[i][0] = corr_val_data[i]
    bb_val_input[i][1] = conf_val_data[i]
    bb_val_input[i][2] = entr_val_data[i]
    bb_val_input[i][3] = m_entr_val_data[i]

print("Blackbox Features Loaded")


att_train_dataset = torch.utils.data.TensorDataset(grad_train_data, bb_train_input, output_train_input, labels_train_input, att_train_label)
att_train_dataloader = torch.utils.data.DataLoader(att_train_dataset, batch_size=128, shuffle=True, num_workers=2)
att_val_dataset = torch.utils.data.TensorDataset(grad_val_data, bb_val_input, output_val_input, labels_val_input, att_val_label)
att_val_dataloader = torch.utils.data.DataLoader(att_val_dataset, batch_size=128, shuffle=False, num_workers=2)
att_test_dataset = torch.utils.data.TensorDataset(grad_test_data, bb_test_input, output_test_input, labels_test_input, att_test_label)
att_test_dataloader = torch.utils.data.DataLoader(att_test_dataset, batch_size=128, shuffle=True, num_workers=2)

# create attack model
class Attack(nn.Module):

    def __init__(self, num_classes=2):
        super(Attack, self).__init__()
        self.grad = nn.Sequential(
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
        self.bb = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(100,64),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(100,64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, g, b, o, l):
        out_g = self.grad(g)
        out_g = out_g.view(out_g.size(0), -1)
        out_b = self.bb(b)
        out_o = self.output(o)
        out_l = self.labels(l)
        _outs= torch.cat((out_g,out_b, out_o, out_l), 1)
        _outs = self.combine(_outs)
        return _outs
    
attack = Attack()
attack = nn.DataParallel(attack).cuda()

# set optimizer
optimizer = optim.Adam(attack.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

best_acc = 0
PATH = './attackmod_net5.pth'

# train attack model
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(att_train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        input_grad, input_bb, input_output, input_labels, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = attack(input_grad.float().cuda(), input_bb.float().cuda(), input_output.float().cuda(),input_labels.float().cuda())
        loss = criterion(outputs, labels.cuda())
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
                for i, data in enumerate(att_val_dataloader, 0):
                    input_grad, input_bb, input_output, input_labels, labels = data
                    outputs = attack(input_grad.float().cuda(), input_bb.float().cuda(), input_output.float().cuda(),input_labels.float().cuda())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.cuda()).sum().item()
            print('Accuracy of the network on the test images: {acc:.3f}'.format(acc=100*correct/total))
            acc=100*correct/total
            if acc > best_acc:
                best_acc = acc
                print(best_acc)
                torch.save(attack.state_dict(), PATH)

            attack.train()


print('Finished Training')

# load best model
attack.load_state_dict(torch.load(PATH))

# evaluate attack mode
attack.eval()

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(att_train_dataloader, 0):
        input_grad, input_bb, input_output, input_labels, labels = data
        outputs = attack(input_grad.float().cuda(), input_bb.float().cuda(), input_output.float().cuda(),input_labels.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the train images: {acc:.3f}'.format(acc=100*correct/total))

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(att_val_dataloader, 0):
        input_grad, input_bb, input_output, input_labels, labels = data
        outputs = attack(input_grad.float().cuda(), input_bb.float().cuda(), input_output.float().cuda(),input_labels.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the val images: {acc:.3f}'.format(acc=100*correct/total))

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(att_test_dataloader, 0):
        input_grad, input_bb, input_output, input_labels, labels = data
        outputs = attack(input_grad.float().cuda(), input_bb.float().cuda(), input_output.float().cuda(),input_labels.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the test images: {acc:.3f}'.format(acc=100*correct/total))