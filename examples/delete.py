#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

from torchknickknacks import modelutils

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

# Delete the last layer of the classifier of the AlexNet model 
model.classifier = modelutils.delete_layers(model.classifier, del_ids = [6])

# Delete the last linear layer of an Elman RNN
simple_rnn = nn.Sequential(
    nn.RNN(2, 
           100, 
           1, 
           batch_first = True),
    nn.Linear(100, 10),
)

simple_rnn = modelutils.delete_layers(simple_rnn, del_ids = [1])