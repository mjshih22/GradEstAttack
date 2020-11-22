import torch
import torchvision
import numpy as np

def estgrad(model, criterion, train_loader, test_loader, input_size, batch_size, path, file):
    if(file == False):

        delta = 0.075
        for i, data in enumerate(train_loader, 0):
            if(i % 20 == 0): print("Loading")
            input, label = data
            grad_est = torch.zeros(input.shape)
            
            for j in range(input.shape[1]):
                
                d_vec = torch.zeros(input.shape[1])
                d_vec[j] = delta

                input_plus = input + d_vec
                input_minus = input - d_vec

                output_plus = model(input_plus)
                output_minus = model(input_minus)

                for k in range(output_plus.shape[0]):
                    out_plus = output_plus[[k]]
                    out_minus = output_minus[[k]]
                    lab = label[[k]]

                    loss_plus = criterion(out_plus, lab)
                    loss_minus = criterion(out_minus, lab)

                    dim_est = (loss_plus - loss_minus)/(2*delta)
                    grad_est[k][j] = dim_est
            
            if(i == 0): 
                att_input = grad_est
            else: 
                att_input = torch.cat((att_input, grad_est))

        for i, data in enumerate(test_loader, 0):
            if(i % 20 == 0): print("Loading")
            input, label = data
            grad_est = torch.zeros(input.shape)
            
            for j in range(input.shape[1]):
                
                d_vec = torch.zeros(input.shape[1])
                d_vec[j] = delta

                input_plus = input + d_vec
                input_minus = input - d_vec

                output_plus = model(input_plus)
                output_minus = model(input_minus)

                for k in range(output_plus.shape[0]):
                    out_plus = output_plus[[k]]
                    out_minus = output_minus[[k]]
                    lab = label[[k]]

                    loss_plus = criterion(out_plus, lab)
                    loss_minus = criterion(out_minus, lab)

                    dim_est = (loss_plus - loss_minus)/(2*delta)
                    grad_est[k][j] = dim_est
            
            
            att_input = torch.cat((att_input, grad_est))

        torch.save(att_input, path)

    if(file ==True):
        att_input = torch.load(path)

    # make targets 
    att_label = torch.ones(2*input_size, dtype = torch.int64)
    for i in range(0,input_size):
        att_label[i] = 0

    # create data loader
    att_dataset = torch.utils.data.TensorDataset(att_input, att_label)
    att_dataloader = torch.utils.data.DataLoader(att_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    return att_dataloader