#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torchknickknacks import modelutils

#Load pretrained AlexNet and CIFAR
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 32
testset = torchvision.datasets.CIFAR10(root = '/Users/alexandrosgoulas/Data/work-stuff/python-code/datasets/CIFAR_torchvision', 
                                       train = False,
                                       download = True, 
                                       transform = transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size = batch_size,
                                         shuffle = True)
# Assign a recorder to a layer of AlexNet:
# here: MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)
layer = list(model.features.named_children())[5][1]
recorder = modelutils.Recorder(layer, record_output = True, backward = False)

#Grab a batch and pass it through the model
X,Y = next(iter(testloader))
out = model(X)

# Compute similarity of representations in selected layer for images in batch
rec = recorder.recording.detach().clone()
rec = rec.reshape(batch_size, -1)
sim = np.corrcoef(rec.numpy())
plt.imshow(sim)
plt.colorbar()