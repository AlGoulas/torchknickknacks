#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

from torchknickknacks import metrics

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

data = torch.rand(64, 3, 224, 224)
output = model(data)
# This is just an example where class coded by 999 has more occurences
# No train test splits are applied to lead to the overrepresentation of class 999 
p = [(1-0.05)/1000]*999
p.append(1-sum(p))
labels = np.random.choice(list(range(1000)), 
                          size = (10000,), 
                          p = p)#imbalanced 1000-class labels
labels = torch.Tensor(labels).long()
weight, label_weight = metrics.class_weights(labels)
loss = torch.nn.CrossEntropyLoss(weight = weight)
l = loss(output, labels[:64])
