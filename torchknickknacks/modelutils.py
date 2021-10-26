#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch 

def get_model_params(model, params_to_get = None):
    '''Extracts the parameters, names, and 'requires gradient' status from a 
    model
    
    Input
    -----
    model: class instance based on the base class torch.nn.Module
    
    params_to_get: list of str, default=None, specifying the names of the 
        parameters to be extracted.
        If None, then all parameters and names of parameters from the model 
        will be extracted
    
    Output
    ------     
    params_name: list, contaning one str for each extracted parameter
    
    params_values: list, containg one tensor corresponding to each 
        parameter. NOTE: The tensor is detached from the computation graph 
    req_grad: list, containing one Boolean variable for each parameter
        denoting the requires_grad status of the tensor/parameter 
        of the model      
    '''    
    params_names = []
    params_values = [] 
    req_grad = []
    for name, param in zip(model.named_parameters(), model.parameters()):             
        if params_to_get is not None:
            if name[0] in params_to_get: 
                params_names.append(name[0])
                params_values.append(param.detach().clone())
                req_grad.append(param.requires_grad)
        else:
            params_names.append(name[0])
            params_values.append(param.detach().clone())
            req_grad.append(param.requires_grad)
                       
    return params_values, params_names, req_grad

def freeze_params(model, 
                  params_to_freeze = None,
                  freeze = True):  
    '''Freeze or unfreeze the parametrs of a model
    
    Input
    -----
    model:  class instance based on the base class torch.nn.Module 
    
    params_to_freeze: list of str specifying the names of the params to be 
        frozen or unfrozen
        
    freeze: bool, default True, specifying the freeze or 
        unfreeze of model params  
        
    Output
    ------
    model: class instance based on the base class torch.nn.Module with changed
        requires_grad param for the anmes params in params_to_freeze
        (freeze = requires_grad is False unfreeze = requires_grad is True)   
    '''
    for name, param in zip(model.named_parameters(), model.parameters()):             
        if params_to_freeze is not None:
            if name[0] in params_to_freeze: 
                param.requires_grad = True if freeze is False else False
        else:
            param.requires_grad = True if freeze is False else False  
    
def delete_layers(model, del_ids = []):
    '''Delete layers from model
    
    Input
    -----
    model: model to be modified
    
    del_ids: list, default [], of int the modules/layers
        that will be deleted
        NOTE: 0, 1... denotes the 1st, 2nd etc layer
        
    Output
    ------ 
    model: model with deleted modules/layers that is an instance of  
        torch.nn.modules.container.Sequential
    '''
    children = [c for i,c in enumerate(model.named_children()) if i not in del_ids]  
    model = torch.nn.Sequential(
        OrderedDict(children)
    ) 
    
    return model

def add_layers(model, modules = []):
    '''Add layers/modules to torch.nn.modules.container.Sequential
    
    Input
    -----
    model: instance of class of base class torch.nn.Module
    
    modules: list of dict
        each dict has key:value pairs
        
        {
        'name': str
        'position': int 
        'module': torch.nn.Module
        }
        
        with: 
            name: str, name to be added in the nn.modules.container.Sequential 
            
            position: int, [0,..N], with N>0, also -1, where N the total
            nr of modules in the torch.nn.modules.container.Sequential
            -1 denotes the module that will be appended at the end
            
            module: torch.nn.Module
    
    Output
    ------
    model: model with added modules/layers that is an instance of   
        torch.nn.modules.container.Sequential
    '''
    all_positions = [m['position'] for m in modules]
    current_children = [c for c in model.named_children()]
    children = []
    children_idx = 0
    iterations = len(current_children) + len(all_positions)
    if -1 in all_positions: iterations -= 1
    for i in range(iterations):
        if i not in all_positions:
            children.append(current_children[children_idx])
            children_idx += 1
        else:
            idx = all_positions.index(i)
            d = modules[idx]
            children.append((d['name'], d['module']))
    if -1 in all_positions:
        idx = all_positions.index(-1)
        d = modules[idx]
        children.append((d['name'], d['module']))
        
    model = torch.nn.Sequential(
        OrderedDict(children)
    ) 

    return model

class Recorder():
    '''Get input, output or parameters to a module/layer 
    by registering forward or backward hooks
    
    Input
    -----
    module: a module of a class in torch.nn.modules 
    
    record_input: bool, default False, deciding if input to module will be
        recorded
        
    record_output: bool, default False, deciding if output to module will be
        recorded 
        
    record_params: bool, default False, deciding if params of module will be
        recorded 
        
    params_to_get: list of str, default None, specifying the parameters to be 
        recorded from the module (if None all parameters are recorded)
        NOTE: meaningful only if record_params
        
    backward: bool, default False, deciding if a forward or backward hook
        will be registered and the recprding will be performed accordingly
        
    custom_fn: function, default None, to be executed in the forward or backward
        pass.
        
        It must have the following signature:
        
        custom_fn(module, output, input, **kwars)
        
        with kwars optional
        
        The signature follows the signature of functions to be registered
        in hooks. See for more details:
        https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
    
     **kwargs: if keyword args are specified they will be passed as to the 
         custom_fn     
         
         
    The attribute recording contains the output, input or params of a module
    '''
    def __init__(self, 
                 module,
                 record_input = False,
                 record_output = False,
                 record_params = False,
                 params_to_get = None,
                 backward = False,
                 custom_fn = None,
                 **kwargs):
        self.params_to_get = params_to_get
        self.kwargs = kwargs if kwargs else None
        if record_input is True:
            fn = self._fn_in 
        elif record_output is True:
            fn = self._fn_out   
        elif record_params is True:
            fn = self._fn_params 
            
        if custom_fn is not None: 
            fn = self._custom_wrapper
            self.custom_fn = custom_fn
            
        if backward is False:
            self.hook = module.register_forward_hook(fn)
        elif backward is True:
            self.hook = module.register_full_backward_hook(fn)
    
    def _fn_in(self, module, input, output):
        self.recording = input
        
    def _fn_out(self, module, input, output):
        self.recording = output
        
    def _fn_params(self, module, input, output):
        params = get_model_params(module, params_to_get = self.params_to_get)[0]
        self.recording = params
        
    def _custom_wrapper(self, module, input, output):
        if self.kwargs: 
            res = self.custom_fn(module, input, output, **self.kwargs)
        else:
            res = self.custom_fn(module, input, output)
        self.recording = res
        
    def close(self):
        self.hook.remove()
