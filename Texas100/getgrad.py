import torch
import torchvision
import numpy as np

def getgrad(model, criterion, train_loader, test_loader, input_size, batch_size = 256):

    # get grad on train data
    for i, data in enumerate(train_loader):
        model.zero_grad()
        input, target = data
        input.requires_grad_(True)
        output = model(input.cuda())

        loss = criterion(output, target.cuda())
        loss.backward()
        g = input.grad.data
        
        if i == 0:
            att_input = g
        else:
            att_input = torch.cat((att_input, g))

    for i, data in enumerate(test_loader):
        model.zero_grad()
        input, target = data
        input.requires_grad_(True)
        output = model(input.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        g = input.grad.data
        att_input = torch.cat((att_input, g))

    # make targets 
    att_label = torch.ones(2*input_size, dtype = torch.int64)
    for i in range(0,input_size):
        att_label[i] = 0

    # create data loader
    att_dataset = torch.utils.data.TensorDataset(att_input, att_label)
    att_dataloader = torch.utils.data.DataLoader(att_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    return att_dataloader