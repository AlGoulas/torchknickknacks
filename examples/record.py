#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torchknickknacks import modelutils

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)

# Register a recorder to the 4th layer of the features part of AlexNet
# Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
# and record the output of the layer during the forward pass
layer = list(model.features.named_children())[3][1]
recorder = modelutils.Recorder(layer, record_output = True, backward = False)
data = torch.rand(64, 3, 224, 224)
output = model(data)
print(recorder.recording)#tensor of shape (64, 192, 27, 27)
recorder.close()#remove the recorder

# Record input to the layer during the forward pass 
recorder = modelutils.Recorder(layer, record_input = True, backward = False)
data = torch.rand(64, 3, 224, 224)
output = model(data)
print(recorder.recording)#tensor of shape (64, 64, 27, 27)
recorder.close()#remove the recorder

# Register a recorder to the 4th layer of the features part of AlexNet
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# and record the output of the layer in the bacward pass 
layer = list(model.features.named_children())[2][1]
# Record output to the layer during the backward pass 
recorder = modelutils.Recorder(layer, record_output = True, backward = True)
data = torch.rand(64, 3, 224, 224)
output = model(data)
loss = torch.nn.CrossEntropyLoss()
labels = torch.randint(1000, (64,))#random labels just to compute a bacward pass
l = loss(output, labels)
l.backward()
print(recorder.recording[0])#tensor of shape (64, 64, 27, 27)
recorder.close()#remove the recorder

# Register a recorder to the 4th layer of the features part of AlexNet
# Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
# and record the parameters of the layer in the forward pass 
layer = list(model.features.named_children())[3][1] 
recorder = modelutils.Recorder(layer, record_params = True, backward = False)
data = torch.rand(64, 3, 224, 224)
output = model(data)
print(recorder.recording)#list of tensors of shape (192, 64, 5, 5) (weights) (192,) (biases) 
recorder.close()#remove the recorder

# A custom function can also be passed to the recorder and perform arbitrary 
# operations. In the example below, the custom function prints the kwargs that 
# are passed along with the custon function and also return 1 (stored in the recorder)
def custom_fn(*args, **kwargs):#signature of any custom fn
    print('custom called')
    for k,v in kwargs.items():
        print('\nkey argument:', k)
        print('\nvalue argument:', v)
    return 1

recorder = modelutils.Recorder(layer, 
                                backward = False, 
                                custom_fn = custom_fn, 
                                print_value = 5)
data = torch.rand(64, 3, 224, 224)
output = model(data)
print(recorder.recording)#list of tensors of shape (192, 64, 5, 5) (weights) (192,) (biases) 
recorder.close()#remove the recorder

# Record output to the layer during the forward pass and store it in folder 
layer = list(model.features.named_children())[3][1]
recorder = modelutils.Recorder(
    layer, 
    record_params = True, 
    backward = False, 
    save_to = '/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/test_recorder'#create the folder before running this example!
)
for _ in range(5):#5 passes e.g. batches, thus 5 stored "recorded" tensors
    data = torch.rand(64, 3, 224, 224)
    output = model(data)
recorder.close()#remove the recorder