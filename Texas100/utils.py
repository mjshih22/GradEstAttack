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
            outputs = model.forward(inputs.cuda()) 
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

def prepare_texas_data(batch_size=100):
    
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
    shadow_indices = r[:train_len//2]
    target_indices = np.delete(np.arange(train_len), shadow_indices)

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]


    test_len = 1*train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len//2]
    target_indices = np.delete(np.arange(test_len), shadow_indices)

    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]


    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    
    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)

    target_train = tensor_data_create(target_train_data, target_train_label)

    target_test = tensor_data_create(target_test_data, target_test_label)

    print('Data loading finished')
    return shadow_train, shadow_test, target_train, target_test

def entr_comp(prediction):
    entr = 0
    for num in prediction:
        if num != 0:
            entr += -1*num*np.log(num)
    return entr
    
def m_entr_comp(prediction, label):
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
    
def thre_setting(tr_values, te_values):
    value_list = np.concatenate((tr_values, te_values))
    thre, max_acc = 0, 0
    for value in value_list:
        tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
        te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
        acc = 0.5*(tr_ratio + te_ratio)
        if acc > max_acc:
            thre, max_acc = value, acc
    return thre

def mem_inf_thre(num_classes, s_tr_values, s_te_values, t_tr_values, t_te_values, s_tr_labels,
                   s_te_labels, t_tr_labels, t_te_labels):
    # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
    # (negative) prediction entropy, and (negative) modified entropy
    predicted = torch.ones(5000)
    thresholds = np.zeros(100)
    for num in range(num_classes):
        thre = thre_setting(s_tr_values[s_tr_labels==num], s_te_values[s_te_labels==num])
        thresholds[num] = thre

    for i in range(0,2500):
        if(t_tr_values[i]>=thresholds[t_tr_labels[i]]):
                predicted[i] = 0
    for i in range(0,2500):
        if(t_te_values[i]>=thresholds[t_te_labels[i]]):
            predicted[2500+i] = 0
    return predicted
