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
grad_train_loader = torch.utils.data.DataLoader(trainset, batch_size=10000,
                                          shuffle=False, num_workers=2)
grad_train_enum = enumerate(grad_train_loader)
batch_idx, (train_data, train_targets) = next(grad_train_enum)


grad_test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=2)
grad_test_enum = enumerate(grad_test_loader)
batch_idx, (test_data, test_targets) = next(grad_test_enum)

# take first 8K and use as attack train data
grad_train_train_data = train_data[0:8000]
grad_train_train_target = train_targets[0:8000]
grad_train_test_data = test_data[0:8000]
grad_train_test_target = test_targets[0:8000]

grad_train_data = torch.cat((grad_train_train_data, grad_train_test_data))
grad_train_target = torch.cat((grad_train_train_target, grad_train_test_target))

# get grad train data
grad_train_input = torch.zeros(16000,3,32,32)

for i in range(0, 16000):
    alexnet.zero_grad()
    input = grad_train_data[[i]]
    input.requires_grad_(True)
    output = alexnet(input)
    target = grad_train_target[[i]]

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    
    grad_train_input[i] = g

# make targets 
grad_train_label = torch.ones(16000, dtype = torch.int64)
for i in range(0,8000):
    grad_train_label[i] = 0

# create data loader
grad_train_dataset = torch.utils.data.TensorDataset(grad_train_input, grad_train_label)
grad_train_dataloader = torch.utils.data.DataLoader(grad_train_dataset, batch_size=250,
                                          shuffle=False, num_workers=2)

# set up test data
grad_test_train_data = train_data[8000:10000]
grad_test_train_target = train_targets[8000:10000]
grad_test_test_data = test_data[8000:10000]
grad_test_test_target = test_targets[8000:10000]

grad_test_data = torch.cat((grad_test_train_data, grad_test_test_data))
grad_test_target = torch.cat((grad_test_train_target, grad_test_test_target))


grad_test_input = torch.zeros(4000,3,32,32)

for i in range(0, 4000):
    alexnet.zero_grad()
    input = grad_test_data[[i]]
    input.requires_grad_(True)
    output = alexnet(input)
    target = grad_test_target[[i]]

    loss = criterion(output, target)
    loss.backward()
    g = input.grad.data
    
    grad_test_input[i] = g

grad_test_label = torch.ones(4000, dtype = torch.int64)

for i in range(0,2000):
    grad_test_label[i] = 0

grad_test_dataset = torch.utils.data.TensorDataset(grad_test_input, grad_test_label)
grad_test_dataloader = torch.utils.data.DataLoader(grad_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

#get correctness, 0 = correct
# also get output and labels
corr_train_data = torch.ones(16000)
labels_train_input = torch.zeros(16000, 100)

for i in range(0, 16000):
    alexnet.zero_grad()
    input = grad_train_data[[i]]
    output = alexnet(input)
    if i == 0:
        output_train_input = output.data
    else:
        output_train_input = torch.cat((output_train_input, output.data))
    _, predicted = torch.max(output.data, 1)
    target = grad_train_target[[i]]
    labels_train_input[i][target] = 1
    if(predicted == target):
        corr_train_data[i] = 0

corr_test_data = torch.ones(4000)
labels_test_input = torch.zeros(4000, 100)

for i in range(0, 4000):
    alexnet.zero_grad()
    input = grad_test_data[[i]]
    output = alexnet(input)
    if i == 0:
        output_test_input = output.data
    else:
        output_test_input = torch.cat((output_test_input, output.data))
    _, predicted = torch.max(output.data, 1)
    target = grad_test_target[[i]]
    labels_test_input[i][target] = 1
    if(predicted == target):
        corr_test_data[i] = 0

print(output_train_input.size())
print(output_test_input.size())
print(labels_train_input.size())
print(labels_test_input.size())

# create data loaders
shadow_train_dataset = torch.utils.data.TensorDataset(grad_train_train_data, grad_train_train_target)
shadow_train_dataloader = torch.utils.data.DataLoader(shadow_train_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

shadow_test_dataset = torch.utils.data.TensorDataset(grad_train_test_data, grad_train_test_target)
shadow_test_dataloader = torch.utils.data.DataLoader(shadow_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

target_train_dataset = torch.utils.data.TensorDataset(grad_test_train_data, grad_test_train_target)
target_train_dataloader = torch.utils.data.DataLoader(target_train_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

target_test_dataset = torch.utils.data.TensorDataset(grad_test_test_data, grad_test_test_target)
target_test_dataloader = torch.utils.data.DataLoader(target_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

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
conf_train_data = np.concatenate((s_tr_conf, s_te_conf))
conf_train_data = torch.from_numpy(conf_train_data)

t_tr_conf = np.array([t_tr_outputs[i, t_tr_labels[i]] for i in range(len(t_tr_labels))])
t_te_conf = np.array([t_te_outputs[i, t_te_labels[i]] for i in range(len(t_te_labels))])
conf_test_data = np.concatenate((t_tr_conf, t_te_conf))
conf_test_data = torch.from_numpy(conf_test_data)

s_tr_entr = np.array([_entr_comp(s_tr_outputs[i]) for i in range(len(s_tr_labels))])
s_te_entr = np.array([_entr_comp(s_te_outputs[i]) for i in range(len(s_te_labels))])
entr_train_data = np.concatenate((s_tr_entr, s_te_entr))
entr_train_data = torch.from_numpy(entr_train_data)

t_tr_entr = np.array([_entr_comp(t_tr_outputs[i]) for i in range(len(t_tr_labels))])
t_te_entr = np.array([_entr_comp(t_te_outputs[i]) for i in range(len(t_te_labels))])
entr_test_data = np.concatenate((t_tr_entr, t_te_entr))
entr_test_data = torch.from_numpy(entr_test_data)

s_tr_m_entr = np.array([_m_entr_comp(s_tr_outputs[i], s_tr_labels[i]) for i in range(len(s_tr_labels))])
s_te_m_entr = np.array([_m_entr_comp(s_te_outputs[i], s_te_labels[i]) for i in range(len(s_te_labels))])
m_entr_train_data = np.concatenate((s_tr_m_entr, s_te_m_entr))
m_entr_train_data = torch.from_numpy(m_entr_train_data)

t_tr_m_entr = np.array([_m_entr_comp(t_tr_outputs[i], t_tr_labels[i]) for i in range(len(t_tr_labels))])
t_te_m_entr = np.array([_m_entr_comp(t_te_outputs[i], t_te_labels[i]) for i in range(len(t_te_labels))])
m_entr_test_data = np.concatenate((t_tr_m_entr, t_te_m_entr))
m_entr_test_data = torch.from_numpy(m_entr_test_data)
  
# create blackbox linear input
bb_train_input = torch.zeros(16000, 4)
for i in range(0,16000):
    bb_train_input[i][0] = corr_train_data[i]
    bb_train_input[i][1] = conf_train_data[i]
    bb_train_input[i][2] = entr_train_data[i]
    bb_train_input[i][3] = m_entr_train_data[i]

bb_test_input = torch.zeros(4000, 4)
for i in range(0,4000):
    bb_test_input[i][0] = corr_test_data[i]
    bb_test_input[i][1] = conf_test_data[i]
    bb_test_input[i][2] = entr_test_data[i]
    bb_test_input[i][3] = m_entr_test_data[i]

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
            nn.Linear(448, 256),
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

# set optimizer
import torch.optim as optim

optimizer = optim.Adam(attack.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)


# train attack model
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(grad_train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        input_grad, labels = data
        input_bb = bb_train_input.narrow(0, i*250, 250)
        input_output = output_train_input.narrow(0,i*250,250)
        input_labels = labels_train_input.narrow(0,i*250,250)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = attack(input_grad, input_bb, input_output,input_labels)
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
                for i, data in enumerate(grad_test_dataloader, 0):
                    input_grad, labels = data
                    input_bb = bb_test_input.narrow(0, i*4, 4)
                    input_output = output_test_input.narrow(0,i*4,4)
                    input_labels = labels_test_input.narrow(0,i*4,4)
                    outputs = attack(input_grad, input_bb, input_output, input_labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: {acc:.3f}'.format(acc=100*correct/total))

            attack.train()


print('Finished Training')

PATH = './attackmod_net.pth'
torch.save(attack.state_dict(), PATH)

# evaluate attack mode
attack.eval()

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(grad_test_dataloader, 0):
        input_grad, labels = data
        input_bb = bb_test_input.narrow(0, i*4, 4)
        input_output = output_test_input.narrow(0,i*4,4)
        input_labels = labels_test_input.narrow(0,i*4,4)
        outputs = attack(input_grad, input_bb, input_output, input_labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: {acc:.3f}'.format(acc=100*correct/total))