#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torchknickknacks import modelutils

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

# Freeze all parameters
modelutils.freeze_params(model, 
                         freeze = True)

# Unfreeze all parameters
modelutils.freeze_params(model, 
                         freeze = False)

# Freeze specific parameters by naming them
params_to_freeze = ['features.0.weight', 'classifier.1.weight']
modelutils.freeze_params(model, 
                         params_to_freeze = params_to_freeze, 
                         freeze = True)

# Unfreeze specific parameters by naming them
params_to_freeze = ['features.0.weight', 'classifier.1.weight']
modelutils.freeze_params(model, 
                         params_to_freeze = params_to_freeze, 
                         freeze = False)