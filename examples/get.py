#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torchknickknacks import modelutils

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

# Get all parameters
params_values, params_names, req_grad = modelutils.get_model_params(model)

# Get only a subset of parameters by passing a list of named parameters
params_to_get = ['features.0.weight', 'classifier.1.weight']
params_values, params_names, req_grad = modelutils.get_model_params(model,
                                                                    params_to_get = params_to_get)