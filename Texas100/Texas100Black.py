#!/usr/bin/env python
# coding: utf-8



import torch
from utils import *
from texas import texas
from membership_inference_attacks import black_box_benchmarks

# load pretrained model
target = texas(num_classes=100)
att_train_train, att_train_test, target_train, target_test = prepare_texas_data(100)

test_size = 5000
split_size = int(test_size/2)

torch.manual_seed(42)
att_val_train, att_test_train = torch.utils.data.random_split(target_train, [split_size, split_size])
torch.manual_seed(torch.initial_seed())
att_val_test, att_test_test = torch.utils.data.random_split(target_test, [split_size, split_size])

att_train_train_loader = torch.utils.data.DataLoader(att_train_train, batch_size=100, shuffle=True, num_workers=2)
att_train_test_loader = torch.utils.data.DataLoader(att_train_test, batch_size=100, shuffle=True, num_workers=2)
# att_val_train_loader = torch.utils.data.DataLoader(att_val_train, batch_size=100, shuffle=False, num_workers=2)
# att_val_test_loader = torch.utils.data.DataLoader(att_val_test, batch_size=100, shuffle=False, num_workers=2)
att_test_train_loader =  torch.utils.data.DataLoader(att_test_train, batch_size=100, shuffle=True, num_workers=2)
att_test_test_loader = torch.utils.data.DataLoader(att_test_test, batch_size=100, shuffle=True, num_workers=2)

checkpoint = torch.load('texas_advreg')
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = k
    else:
        k = k.replace('module.', '')
    new_state_dict[k] = v

target.load_state_dict(new_state_dict)
target.eval()

shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(target, att_train_train_loader, att_train_test_loader, 
                              target, att_test_train_loader, att_test_test_loader)
    
print('Perform membership inference attacks!!!')
MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                    target_train_performance,target_test_performance,num_classes=100)
MIA._mem_inf_benchmarks()