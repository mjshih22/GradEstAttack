import torch
import torchvision
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from alexnet import alexnet
from densenet import densenet
from getgrad import getgrad
from estgrad import estgrad

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

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=250, shuffle=False, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=250, shuffle=False, num_workers=2)
att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=250, shuffle=False, num_workers=2)
att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=250, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=250, shuffle=False, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=250, shuffle=False, num_workers=2)

target = densenet()
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

# set criterion
criterion = nn.CrossEntropyLoss()


# create data loader with gradients for attack
""" print("Loading Train Data")
att_train_dataloader = estgrad(target, criterion, att_train_train_loader, att_train_test_loader, train_size, 128, './att_train.npy', False) """
print("Loading Val Data")
att_val_dataloader = estgrad(target, criterion, att_val_train_loader, att_val_test_loader, val_size, 256, './att_val.npy', False)
print("Loading Test Data")
att_test_dataloader = estgrad(target, criterion, att_test_train_loader, att_test_test_loader, test_size, 256, './att_test.npy', False)
print("Data has been loaded")

""" # create attack model
attack = alexnet(num_classes = 2)
attack = nn.DataParallel(attack).cuda()

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
        outputs = attack(inputs.float().cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 24 == 23:    # print once per epoch
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 24))
            running_loss = 0.0

	    # evaluate attack mode
            attack.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for data in att_val_dataloader:
                    images, labels = data
                    outputs = attack(images.float().cuda())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.cuda()).sum().item()
            print('Accuracy of the network on the val images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))
            acc=100*correct/total
            if acc > best_acc:
                best_acc = acc
                print(best_acc)
                torch.save(attack.state_dict(), PATH)
            attack.train()

print('Finished Training')

# load best model
attack.load_state_dict(torch.load(PATH))

# evaluate attack model on train and test data
attack.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in att_train_dataloader:
        images, labels = data
        outputs = attack(images.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the train images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))

correct = 0
total = 0
with torch.no_grad():
    for data in att_val_dataloader:
        images, labels = data
        outputs = attack(images.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the validation images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total))

correct = 0
total = 0
with torch.no_grad():
    for data in att_test_dataloader:
        images, labels = data
        outputs = attack(images.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the test images using gradients of inputs: {acc:.3f}'.format(acc=100*correct/total)) """