import torch
import torchvision
import numpy as np

def estgrad(model, criterion, train_loader, test_loader, input_size, batch_size, path, file):
    if(file == False):

        delta = .005
        for i, data in enumerate(train_loader, 0):
            if(i % 10 == 0): print("Loading")
            input, label = data
            grad_est = np.zeros(input.shape)
            
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):

                        d_vec = np.zeros(input.shape[1:])
                        d_vec[j][k][l] = delta

                        input_plus = input.clone().detach()
                        input_plus += torch.from_numpy(d_vec).float()
                        input_minus = input.clone().detach()
                        input_minus -= torch.from_numpy(d_vec).float()

                        output_plus = model(input_plus.cuda())
                        output_minus = model(input_minus.cuda())

                        for m in range(output_plus.shape[0]):
                            out_plus = output_plus[[m]]
                            out_minus = output_minus[[m]]
                            lab = label[[m]]

                            loss_plus = criterion(out_plus, lab.detach().cuda())
                            loss_minus = criterion(out_minus, lab.detach().cuda())

                            dim_est = (loss_plus - loss_minus)/(2*delta)
                            grad_est[m][j][k][l] = dim_est
            
            if(i == 0): 
                att_input = grad_est
            else: 
                att_input = np.concatenate((att_input, grad_est))

        for i, data in enumerate(test_loader, 0):
            if(i % 10 == 0): print("Loading")
            input, label = data
            grad_est = np.zeros(input.shape)
            
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):

                        d_vec = np.zeros(input.shape[1:])
                        d_vec[j][k][l] = delta

                        input_plus = input.clone().detach()
                        input_plus += torch.from_numpy(d_vec).float()
                        input_minus = input.clone().detach()
                        input_minus -= torch.from_numpy(d_vec).float()

                        output_plus = model(input_plus.cuda())
                        output_minus = model(input_minus.cuda())

                        for m in range(output_plus.shape[0]):
                            out_plus = output_plus[[m]]
                            out_minus = output_minus[[m]]
                            lab = label[[m]]

                            loss_plus = criterion(out_plus, lab.detach().cuda())
                            loss_minus = criterion(out_minus, lab.detach().cuda())

                            dim_est = (loss_plus - loss_minus)/(2*delta)
                            grad_est[m][j][k][l] = dim_est
            
            
            att_input = np.concatenate((att_input, grad_est))

        np.save(path, att_input)

    if(file ==True):
        att_input = np.load(path)

    # make targets 
    att_label = torch.ones(2*input_size, dtype = torch.int64)
    for i in range(0,input_size):
        att_label[i] = 0

    att_input = torch.from_numpy(att_input)
    # create data loader
    att_dataset = torch.utils.data.TensorDataset(att_input, att_label)
    att_dataloader = torch.utils.data.DataLoader(att_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return att_dataloader