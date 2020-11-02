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

    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=1)

    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    shadow_test_loader = torch.utils.data.DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=1)

    target_train = tensor_data_create(target_train_data, target_train_label)
    target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=1)

    target_test = tensor_data_create(target_test_data, target_test_label)
    target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Data loading finished')
    return shadow_train_loader, shadow_test_loader, target_train_loader, target_test_loader