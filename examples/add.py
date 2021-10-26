#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

from torchknickknacks import modelutils

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

# Delete the last layer of the classifier of the AlexNet model 
model.classifier = modelutils.delete_layers(model.classifier, del_ids = [6])

# Add back to the model the deleted layer
module = {
         'name': '6',
         'position': 6,
         'module': nn.Linear(in_features = 4096, out_features = 1000, bias = True) 
         }

model.classifier = modelutils.add_layers(model.classifier, modules = [module]) 