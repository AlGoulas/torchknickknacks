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
    
    return model

def delete_layers(model, del_ids = []):
    '''Delete layers from model
    
    Input
    -----
    model: instance of class of base class torch.nn.Module
    
    del_ids: list, default [], of int or str specifying the modules/layers
        that will be deleted
        
    Output
    ------ 
    model: model with deleted modules/layers that is an instance of  
        torch.nn.modules.container.Sequential
    '''
    # Check del_ids int or str?
    what_type = None
    type_str = [isinstance(d, str) for d in del_ids]
    if all(type_str): 
        what_type = 'str' 
    else:
        type_int = [isinstance(d, int) for d in del_ids] 
        if all(type_int):
            what_type = 'int'
    if what_type is None: 
        print('\ndel_ids must be a list of int or a list of str')
        return
    if what_type == 'int':
        children = [c for i,c in enumerate(model.named_children()) if i not in del_ids]
    elif what_type == 'str':
        children = [c for i,c in enumerate(model.named_children()) if c[0] not in del_ids]#check if name of module in the list
        
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