#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torchknickknacks import metrics

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)
data = torch.rand(64, 3, 224, 224)
output = model(data)
labels = torch.randint(1000, (64,))#random labels 
acc = metrics.calc_accuracy(output = output, labels = labels) 