#!/usr/bin/env python
# coding: utf-8


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
from utils import *
from membership_inference_attacks import black_box_benchmarks

# define model
class TexasClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(TexasClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

# load pretrained model
model = TexasClassifier(num_classes=100)
shadow_train_loader, shadow_test_loader, target_train_loader, target_test_loader = prepare_texas_data(100)
checkpoint = torch.load('texas_natural')
state_dict = checkpoint['state_dict']

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = k
    else:
        k = k.replace('module.', '')
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
prepare_model_performance(model, shadow_train_loader, shadow_test_loader, 
                            model, target_train_loader, target_test_loader)
    
print('Perform membership inference attacks!!!')
MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                    target_train_performance,target_test_performance,num_classes=100)
MIA._mem_inf_benchmarks()
