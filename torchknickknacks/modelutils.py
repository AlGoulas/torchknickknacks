#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


    
    
    
