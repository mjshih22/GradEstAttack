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
import torchvision.transforms as transforms

def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

def prepare_texas_data():
    
    DATASET_PATH = './datasets/texas/'
    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    DATASET_FEATURES = os.path.join(DATASET_PATH,'texas/100/feats')
    DATASET_LABELS = os.path.join(DATASET_PATH,'texas/100/labels')
    DATASET_NUMPY = 'data.npz'
    
    if not os.path.isfile(DATASET_FEATURES):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')

        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)
        print('reading dataset...')
        data_set_features =np.genfromtxt(DATASET_FEATURES,delimiter=',')
        data_set_label =np.genfromtxt(DATASET_LABELS,delimiter=',')
        print('finish reading!')

        X =data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)
    
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    r = np.load('./dataset_shuffle/random_r_texas100.npy')
    X=X[r]
    Y=Y[r]

    len_train =len(X)
    train_classifier_ratio, train_attack_ratio = float(10000)/float(X.shape[0]),0.3
    train_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    train_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    np.random.seed(100)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    att_train_indices = r[:8000]
    att_test_indices = r[8000:10000]

    shadow_train_data, shadow_train_label = train_data[att_train_indices], train_label[att_train_indices]
    target_train_data, target_train_label = train_data[att_test_indices], train_label[att_test_indices]

    shadow_test_data, shadow_test_label = test_data[att_train_indices], test_label[att_train_indices]
    target_test_data, target_test_label = test_data[att_test_indices], test_label[att_test_indices]

    print('Data loading finished')
    
    return shadow_train_data, shadow_train_label, target_train_data, target_train_label, \
    shadow_test_data, shadow_test_label, target_test_data, target_test_label

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

# load data
att_train_train_data, att_train_train_target, att_test_train_data, att_test_train_target, \
    att_train_test_data, att_train_test_target, att_test_test_data, att_test_test_target = prepare_texas_data()

# set up attack model training data
att_train_train_data = torch.from_numpy(att_train_train_data)
att_train_train_target = torch.from_numpy(att_train_train_target)

att_train_test_data = torch.from_numpy(att_train_test_data)
att_train_test_target = torch.from_numpy(att_train_test_target)

att_train_data = torch.cat((att_train_train_data, att_train_test_data))
att_train_target = torch.cat((att_train_train_target, att_train_test_target))


# get grad train data
att_train_input = torch.zeros(16000,6169)

for i in range(0, 16000):
    model.zero_grad()
    input = att_train_data[[i]]
    input.requires_grad_(True)
    output = model(input.float())
    target = att_train_target[[i]]

    loss = criterion(output, target.long())
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
                                          shuffle=False, num_workers=2)

# set up attack model test data

att_test_train_data = torch.from_numpy(att_test_train_data)
att_test_train_target = torch.from_numpy(att_test_train_target)

att_test_test_data = torch.from_numpy(att_test_test_data)
att_test_test_target = torch.from_numpy(att_test_test_target)

att_test_data = torch.cat((att_test_train_data, att_test_test_data))
att_test_target = torch.cat((att_test_train_target, att_test_test_target))

# get grad test data
att_test_input = torch.zeros(4000,6169)

for i in range(0, 4000):
    model.zero_grad()
    input = att_test_data[[i]]
    input.requires_grad_(True)
    output = model(input.float())
    target = att_test_target[[i]]

    loss = criterion(output, target.long())
    loss.backward()
    g = input.grad.data
    
    att_test_input[i] = g

# make targets
att_test_label = torch.ones(4000, dtype = torch.int64)
for i in range(0,2000):
    att_test_label[i] = 0

#create dataloader
att_test_dataset = torch.utils.data.TensorDataset(att_test_input, att_test_label)
att_test_dataloader = torch.utils.data.DataLoader(att_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

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
    for data in att_test_dataloader:
        images, labels = data
        outputs = attack(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))