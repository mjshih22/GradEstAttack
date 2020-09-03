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
    predicted = torch.ones(4000)
    thresholds = np.zeros(100)
    for num in range(num_classes):
        thre = _thre_setting(s_tr_values[s_tr_labels==num], s_te_values[s_te_labels==num])
        thresholds[num] = thre

    for i in range(0,2000):
        if(t_tr_values[i]>=thresholds[t_tr_labels[i]]):
                predicted[i] = 0
    for i in range(0,2000):
        if(t_te_values[i]>=thresholds[t_te_labels[i]]):
            predicted[2000+i] = 0
    return predicted

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

# set up attack model test data

att_test_train_data = torch.from_numpy(att_test_train_data)
att_test_train_target = torch.from_numpy(att_test_train_target)

att_test_test_data = torch.from_numpy(att_test_test_data)
att_test_test_target = torch.from_numpy(att_test_test_target)

att_test_data = torch.cat((att_test_train_data, att_test_test_data))
att_test_target = torch.cat((att_test_train_target, att_test_test_target))


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

att_test_label = torch.ones(4000, dtype = torch.int64)
for i in range(0,2000):
    att_test_label[i] = 0

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

PATH = './attack_net.pth'
attack.load_state_dict(torch.load(PATH))

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
predicted_corr = torch.ones(4000)
correct = 0
for i in range(0, 4000):
    model.zero_grad()
    input = att_test_data[[i]]
    output = model(input.float())
    _, predicted = torch.max(output.data, 1)
    target = att_test_target[[i]]
    if(predicted == target):
        predicted_corr[i] = 0

for i in range(0,4000):
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
for i in range(0,4000):
    if(predicted_conf[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using confidence: {acc:.3f}'.format(acc=100*correct/total))

# get prediction of entropy
predicted_entr = _mem_inf_thre(num_classes, -s_tr_entr, -s_te_entr, -t_tr_entr, -t_te_entr, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,4000):
    if(predicted_entr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using entropy: {acc:.3f}'.format(acc=100*correct/total))

# get prediction of modified entropy
predicted_m_entr = _mem_inf_thre(num_classes, -s_tr_m_entr, -s_te_m_entr, -t_tr_m_entr, -t_te_m_entr, s_tr_labels,
                               s_te_labels, t_tr_labels, t_te_labels)
correct = 0
for i in range(0,4000):
    if(predicted_m_entr[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using modified entropy: {acc:.3f}'.format(acc=100*correct/total))

predicted_ens = torch.zeros(4000)
for i in range(0,4000):
    ens = predicted_grad[i] + predicted_corr[i] + predicted_conf[i] + predicted_entr[i] + predicted_m_entr[i]
    if(ens >= 3):
        predicted_ens[i] = 1

correct = 0
for i in range(0,4000):
    if(predicted_ens[i] == att_test_label[i]):
        correct +=1

print('Accuracy of the network on the test images using ensemble: {acc:.3f}'.format(acc=100*correct/total))
