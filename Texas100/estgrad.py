import torch
import torchvision
import numpy as np

def estgrad(model, criterion, train_loader, test_loader, input_size, batch_size, path, file, shuf):
    if(file == False):

        delta = .005
        for i, data in enumerate(train_loader, 0):
            if(i % 20 == 0): print("Loading")
            input, label = data
            grad_est = np.zeros(input.shape)
            
            for j in range(input.shape[1]):
                
                d_vec = np.zeros(input.shape[1])
                d_vec[j] = delta

                input_plus = input.clone().detach()
                input_plus += d_vec
                input_minus = input.clone().detach()
                input_minus -= d_vec

                output_plus = model(input_plus.float().cuda())
                output_minus = model(input_minus.float().cuda())

                for k in range(output_plus.shape[0]):
                    out_plus = output_plus[[k]]
                    out_minus = output_minus[[k]]
                    lab = label[[k]]

                    loss_plus = criterion(out_plus, lab.cuda()).detach()
                    loss_minus = criterion(out_minus, lab.cuda()).detach()

                    dim_est = (loss_plus - loss_minus)/(2*delta)
                    grad_est[k][j] = dim_est
            
            if(i == 0): 
                att_input = grad_est
            else: 
                att_input = np.concatenate((att_input, grad_est))

        for i, data in enumerate(test_loader, 0):
            if(i % 20 == 0): print("Loading")
            input, label = data
            grad_est = np.zeros(input.shape)
            
            for j in range(input.shape[1]):
                
                d_vec = np.zeros(input.shape[1])
                d_vec[j] = delta

                input_plus = input.clone().detach()
                input_plus += d_vec
                input_minus = input.clone().detach()
                input_minus -= d_vec

                output_plus = model(input_plus.float().cuda())
                output_minus = model(input_minus.float().cuda())

                for k in range(output_plus.shape[0]):
                    out_plus = output_plus[[k]]
                    out_minus = output_minus[[k]]
                    lab = label[[k]]

                    loss_plus = criterion(out_plus, lab.cuda()).detach()
                    loss_minus = criterion(out_minus, lab.cuda()).detach()

                    dim_est = (loss_plus - loss_minus)/(2*delta)
                    grad_est[k][j] = dim_est
            
            
            att_input = np.concatenate((att_input, grad_est))

        np.savetxt(path, att_input)

    if(file ==True):
        att_input = np.loadtxt(path)

    # make targets 
    att_label = torch.ones(2*input_size, dtype = torch.int64)
    for i in range(0,input_size):
        att_label[i] = 0

    att_input = torch.from_numpy(att_input)
    # create data loader
    att_dataset = torch.utils.data.TensorDataset(att_input, att_label)
    att_dataloader = torch.utils.data.DataLoader(att_dataset, batch_size=batch_size,
                                            shuffle=shuf, num_workers=2)

    return att_dataloader