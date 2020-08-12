#!/usr/bin/env python
# coding: utf-8



import torch
import torchvision
import numpy as np
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
            predicted[2000+i]
    return predicted

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)


testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
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
        k = k.replace('module.', '')
    new_state_dict[k] = v

alexnet.load_state_dict(new_state_dict)

alexnet.eval()

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
        outputs = attack(images)
        _, pred = torch.max(outputs.data, 1)
        predicted_grad = torch.cat((predicted_grad, pred))
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))


#get prediction of correctness
predicted_corr = torch.ones(4000)
correct = 0
for i in range(0, 4000):
    alexnet.zero_grad()
    input = att_test_data[[i]]
    output = alexnet(input)
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
                                          shuffle=True, num_workers=2)

shadow_test_dataset = torch.utils.data.TensorDataset(att_train_test_data, att_train_test_target)
shadow_test_dataloader = torch.utils.data.DataLoader(shadow_test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

target_train_dataset = torch.utils.data.TensorDataset(att_test_train_data, att_test_train_target)
target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

target_test_dataset = torch.utils.data.TensorDataset(att_test_test_data, att_test_test_target)
target_test_dataloader = torch.utils.data.DataLoader(target_test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

# create blackbox_attacks variables
shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(alexnet, shadow_train_dataloader, shadow_test_dataloader, 
                              alexnet, target_train_dataloader, target_test_dataloader)

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
