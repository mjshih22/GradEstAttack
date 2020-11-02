#!/usr/bin/env python
# coding: utf-8



import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import math


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


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

att_train_train, att_test_train, _ = torch.utils.data.random_split(trainset, [8000,2000, 40000])
att_train_test, att_test_test = torch.utils.data.random_split(testset, [8000,2000])

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=100, shuffle=False, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=100, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=100, shuffle=False, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=100, shuffle=False, num_workers=2)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

resnet = ResNet(depth = 164, block_name='bottleNeck')
resnet = nn.DataParallel(resnet).cuda()

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

resnet.load_state_dict(new_state_dict)

resnet.eval()

print("Model has been loaded\n")

# set criterion
criterion = nn.CrossEntropyLoss()

# get grad on train data
for i, data in enumerate(att_train_train_loader):
    resnet.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = resnet(input.cuda())

    loss = criterion(output, target.cuda())
    loss.backward()
    g = input.grad.data
    
    if i == 0:
        grad_train_data = input
        grad_train_target = target
        grad_train_input = g
    else:
        grad_train_data = torch.cat((grad_train_data, input))
        grad_train_target = torch.cat((grad_train_target, target))
        grad_train_input = torch.cat((grad_train_input, g))
        
for i, data in enumerate(att_train_test_loader):
    resnet.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = resnet(input.cuda())

    loss = criterion(output, target.cuda())
    loss.backward()
    g = input.grad.data

    grad_train_data = torch.cat((grad_train_data, input))
    grad_train_target = torch.cat((grad_train_target, target))
    grad_train_input = torch.cat((grad_train_input, g))

# make targets 
grad_train_label = torch.ones(16000, dtype = torch.int64)
for i in range(0,8000):
    grad_train_label[i] = 0

# create data loader
grad_train_dataset = torch.utils.data.TensorDataset(grad_train_input, grad_train_label)
grad_train_dataloader = torch.utils.data.DataLoader(grad_train_dataset, batch_size=250,
                                          shuffle=True, num_workers=2)

# get grad on test data
for i, data in enumerate(att_test_train_loader):
    resnet.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = resnet(input.cuda())

    loss = criterion(output, target.cuda())
    loss.backward()
    g = input.grad.data
    
    if i == 0:
        grad_test_data = input
        grad_test_target = target
        grad_test_input = g
    else:
        grad_test_data = torch.cat((grad_test_data, input))
        grad_test_target = torch.cat((grad_test_target, target))
        grad_test_input = torch.cat((grad_test_input, g))
        
for i, data in enumerate(att_test_test_loader):
    resnet.zero_grad()
    input, target = data
    input.requires_grad_(True)
    output = resnet(input.cuda())

    loss = criterion(output, target.cuda())
    loss.backward()
    g = input.grad.data
    grad_test_data = torch.cat((grad_test_data, input))
    grad_test_target = torch.cat((grad_test_target, target))
    grad_test_input = torch.cat((grad_test_input, g))

# make targets 
grad_test_label = torch.ones(4000, dtype = torch.int64)
for i in range(0,2000):
    grad_test_label[i] = 0

# create data loader
grad_test_dataset = torch.utils.data.TensorDataset(grad_test_input, grad_test_label)
grad_test_dataloader = torch.utils.data.DataLoader(grad_test_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

print("Grad data has been loaded\n")

print(grad_train_data.shape)
print(grad_train_target.shape)
print(grad_test_data.shape)
print(grad_test_target.shape)


#get correctness, 0 = correct
# also get output and labels
corr_train_data = torch.ones(16000)
labels_train_input = torch.zeros(16000, 100)

for i, data in enumerate(att_train_train_loader):
    resnet.zero_grad()
    input, target = data
    output = resnet(input.cuda())
    _, predicted = torch.max(output.data, 1)

    soft = torch.zeros((100,100))

    for j in range(100):
        exp = np.exp(output.data[0].cpu())
        total = torch.sum(exp)
        soft[0] = exp/total

    if i == 0:
        output_train_input = soft.cuda()
        target_train = target
        predicted_train = predicted
    else:
        output_train_input = torch.cat((output_train_input, soft.cuda()))
        target_train = torch.cat((target_train, target))
        predicted_train = torch.cat((predicted_train, predicted))

print(torch.sum(output_train_input[0]))

for i, data in enumerate(att_train_test_loader):
    resnet.zero_grad()
    input, target = data
    output = resnet(input.cuda())
    _, predicted = torch.max(output.data, 1)

    soft = torch.zeros((100,100))

    for j in range(100):
        exp = np.exp(output.data[0].cpu())
        total = torch.sum(exp)
        soft[0] = exp/total
    
    output_train_input = torch.cat((output_train_input, soft.cuda()))
    target_train = torch.cat((target_train, target))
    predicted_train = torch.cat((predicted_train, predicted))

for i in range(16000):
    target = target_train[[i]]
    predicted = predicted_train[[i]]

    labels_train_input[i][target] = 1
    if(predicted == target.cuda()):
            corr_train_data[i] = 0


corr_test_data = torch.ones(4000)
labels_test_input = torch.zeros(4000, 100)

for i, data in enumerate(att_test_train_loader):
    resnet.zero_grad()
    input, target = data
    output = resnet(input.cuda())
    _, predicted = torch.max(output.data, 1)

    soft = torch.zeros((100,100))

    for j in range(100):
        exp = np.exp(output.data[0].cpu())
        total = torch.sum(exp)
        soft[0] = exp/total

    if i == 0:
        output_test_input = soft.cuda()
        target_test = target
        predicted_test = predicted
    else:
        output_test_input = torch.cat((output_test_input, soft.cuda()))
        target_test = torch.cat((target_test, target))
        predicted_test = torch.cat((predicted_test, predicted))

print(torch.sum(output_test_input[0]))

for i, data in enumerate(att_test_test_loader):
    resnet.zero_grad()
    input, target = data
    output = resnet(input.cuda())
    
    soft = torch.zeros((100,100))

    for j in range(100):
        exp = np.exp(output.data[0].cpu())
        total = torch.sum(exp)
        soft[0] = exp/total

    _, predicted = torch.max(output.data, 1)
    
    output_test_input = torch.cat((output_test_input, soft.cuda()))
    target_test = torch.cat((target_test, target))
    predicted_test = torch.cat((predicted_test, predicted))


for i in range(4000):
    target = target_test[[i]]
    predicted = predicted_test[[i]]

    labels_test_input[i][target] = 1
    if(predicted == target.cuda()):
            corr_test_data[i] = 0


# create blackbox_attacks variables
shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(resnet, att_train_train_loader, att_train_test_loader, 
                              resnet, att_test_train_loader, att_test_test_loader)

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

print("bb_train_input")
print(bb_train_input[0])

bb_test_input = torch.zeros(4000, 4)
for i in range(0,4000):
    bb_test_input[i][0] = corr_test_data[i]
    bb_test_input[i][1] = conf_test_data[i]
    bb_test_input[i][2] = entr_test_data[i]
    bb_test_input[i][3] = m_entr_test_data[i]

print("bb_test_input")
print(bb_test_input[0])

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
        outputs = attack(input_grad.cuda(), input_bb.cuda(), input_output.cuda(),input_labels.cuda())
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
                for i, data in enumerate(grad_test_dataloader, 0):
                    input_grad, labels = data
                    input_bb = bb_test_input.narrow(0, i*4, 4)
                    input_output = output_test_input.narrow(0,i*4,4)
                    input_labels = labels_test_input.narrow(0,i*4,4)
                    outputs = attack(input_grad.cuda(), input_bb.cuda(), input_output.cuda(), input_labels.cuda())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.cuda()).sum().item()

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
        outputs = attack(input_grad.cuda(), input_bb.cuda(), input_output.cuda(), input_labels.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the test images: {acc:.3f}'.format(acc=100*correct/total))