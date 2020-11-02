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
            outputs = model.forward(inputs) 
            return_outputs.append( softmax_by_row(outputs.data.cpu().numpy()) )
        return_outputs = np.concatenate(return_outputs)
        return_labels = np.concatenate(return_labels)
        return (return_outputs, return_labels)
    
    shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
    shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)
    
    target_train_performance = _model_predictions(target_model, target_train_loader)
    target_test_performance = _model_predictions(target_model, target_test_loader)
    return shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance

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
    predicted = torch.ones(19732)
    thresholds = np.zeros(100)
    for num in range(num_classes):
        thre = _thre_setting(s_tr_values[s_tr_labels==num], s_te_values[s_te_labels==num])
        thresholds[num] = thre

    for i in range(0,9866):
        if(t_tr_values[i]>=thresholds[t_tr_labels[i]]):
                predicted[i] = 0
    for i in range(0,9866):
        if(t_te_values[i]>=thresholds[t_te_labels[i]]):
            predicted[9866+i] = 0
    return predicted

def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

def prepare_model_performance(shadow_model, shadow_train_loader, shadow_test_loader,
                              target_model, target_train_loader, target_test_loader):
    def _model_predictions(model, dataloader):
        return_outputs, return_labels = [], []

        for (inputs, labels) in dataloader:
            return_labels.append(labels.numpy())
            outputs = model.forward(inputs.float()) 
            return_outputs.append( softmax_by_row(outputs.data.cpu().numpy()) )
        return_outputs = np.concatenate(return_outputs)
        return_labels = np.concatenate(return_labels)
        return (return_outputs, return_labels)
    
    shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
    shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)
    
    target_train_performance = _model_predictions(target_model, target_train_loader)
    target_test_performance = _model_predictions(target_model, target_test_loader)
    return shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance

def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

def prepare_purchase_data(batch_size=250):
    DATASET_PATH='./datasets/purchase'
    DATASET_NAME= 'dataset_purchase'
    DATASET_NUMPY = 'data.npz'

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    
    DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)
    
    if not os.path.isfile(DATASET_FILE):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)
    
        print('reading dataset...')
        data_set =np.genfromtxt(DATASET_FILE,delimiter=',')
        print('finish reading!')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)
    
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    len_train =len(X)
    r = np.load('./dataset_shuffle/random_r_purchase100.npy')

    X=X[r]
    Y=Y[r]
        
    train_classifier_ratio, train_attack_ratio = 0.1,0.3
    train_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    train_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    np.random.seed(100)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len//2]
    target_indices = r[train_len//2:]

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    test_len = 1*train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len//2]
    target_indices = r[test_len//2:]
    
    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    print('Data loading finished')
    
    return shadow_train_data, shadow_train_label, target_train_data, target_train_label, \
    shadow_test_data, shadow_test_label, target_test_data, target_test_label

# define model
class PurchaseClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600,1024),
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
model = PurchaseClassifier(num_classes=100)
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
model.load_state_dict(new_state_dict)

model.eval()

# set criterion
criterion = nn.CrossEntropyLoss()

test_size = 19732

# load data
att_train_train_data, att_train_train_target, att_test_train_data, att_test_train_target, \
    att_train_test_data, att_train_test_target, att_test_test_data, att_test_test_target = prepare_purchase_data()

# set up attack model training data
att_train_train_data = torch.from_numpy(att_train_train_data)
att_train_train_target = torch.from_numpy(att_train_train_target)

att_train_test_data = torch.from_numpy(att_train_test_data)
att_train_test_target = torch.from_numpy(att_train_test_target)

att_train_data = torch.cat((att_train_train_data, att_train_test_data))
att_train_target = torch.cat((att_train_train_target, att_train_test_target))

# get grad train data
att_train_input = torch.zeros(test_size,600)

for i in range(0, test_size):
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
att_train_label = torch.ones(test_size, dtype = torch.int64)
for i in range(0,int(test_size/2)):
    att_train_label[i] = 0

# create data loader
att_train_dataset = torch.utils.data.TensorDataset(att_train_input, att_train_label)
att_train_dataloader = torch.utils.data.DataLoader(att_train_dataset, batch_size=256,
                                          shuffle=True, num_workers=2)


# set up attack model test data
att_test_train_data = torch.from_numpy(att_test_train_data)
att_test_train_target = torch.from_numpy(att_test_train_target)

att_test_test_data = torch.from_numpy(att_test_test_data)
att_test_test_target = torch.from_numpy(att_test_test_target)

att_test_data = torch.cat((att_test_train_data, att_test_test_data))
att_test_target = torch.cat((att_test_train_target, att_test_test_target))

# get grad test data
att_test_input = torch.zeros(test_size,600)

for i in range(0, test_size):
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
att_test_label = torch.ones(test_size, dtype = torch.int64)
for i in range(0,int(test_size/2)):
    att_test_label[i] = 0

#create dataloader
att_test_dataset = torch.utils.data.TensorDataset(att_test_input, att_test_label)
att_test_dataloader = torch.utils.data.DataLoader(att_test_dataset, batch_size=256,
                                          shuffle=False, num_workers=2)

# create attack model
class Attack(nn.Module):
    def __init__(self,num_classes=2):
        super(Attack, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600,1024),
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
predicted_corr = torch.ones(test_size)
correct = 0
for i in range(0, test_size):
    model.zero_grad()
    input = att_test_data[[i]]
    output = model(input.float())
    _, predicted = torch.max(output.data, 1)
    target = att_test_target[[i]]
    if(predicted == target):
        predicted_corr[i] = 0

for i in range(0,test_size):
    if(predicted_corr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using correctness: {acc:.3f}'.format(acc=100*correct/total))

# create data loaders
shadow_train_dataset = torch.utils.data.TensorDataset(att_train_train_data, att_train_train_target)
shadow_train_dataloader = torch.utils.data.DataLoader(shadow_train_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

shadow_test_dataset = torch.utils.data.TensorDataset(att_train_test_data, att_train_test_target)
shadow_test_dataloader = torch.utils.data.DataLoader(shadow_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

target_train_dataset = torch.utils.data.TensorDataset(att_test_train_data, att_test_train_target)
target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

target_test_dataset = torch.utils.data.TensorDataset(att_test_test_data, att_test_test_target)
target_test_dataloader = torch.utils.data.DataLoader(target_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

# create blackbox_attacks variables
shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(model, shadow_train_dataloader, shadow_test_dataloader, 
                              model, target_train_dataloader, target_test_dataloader)

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
    ens = 3*predicted_grad[i] + predicted_corr[i] + predicted_conf[i] + predicted_entr[i] + predicted_m_entr[i]
    if(ens/7 >=.5):
        predicted_ens[i] = 1

correct = 0
for i in range(0,test_size):
    if(predicted_ens[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using ensemble: {acc:.3f}'.format(acc=100*correct/total))
