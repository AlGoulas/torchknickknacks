#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict
from functools import partial
from pathlib import Path
import pickle 

import torch 

def get_model_params(model, params_to_get = None, detach = True):
    '''Extracts the parameters, names, and 'requires gradient' status from a 
    model
    
    Input
    -----
    model: class instance based on the base class torch.nn.Module
    
    params_to_get: list of str, default=None, specifying the names of the 
        parameters to be extracted
        If None, then all parameters and names of parameters from the model 
        will be extracted
        
    detach: bool, default True, detach the tensor from the computational graph    
    
    Output
    ------     
    params_name: list, contaning one str for each extracted parameter
    
    params_values: list, containg one tensor corresponding to each 
        parameter. 
        NOTE: The tensor is detached from the computation graph 
        
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
                if detach is True:
                    params_values.append(param.detach().clone())
                elif detach is False:
                    params_values.append(param.clone())
                req_grad.append(param.requires_grad)
        else:
            params_names.append(name[0])
            if detach is True:
                params_values.append(param.detach().clone())
            elif detach is False:
                params_values.append(param.clone())
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
    
     save_to: str, default None, specifying a path to a folder for all recordings
         to be saved.
         NOTE: recodrings are saved with filename: recording_0, recording_1, recording_N
         
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
                 save_to = None,
                 **kwargs):
        self.params_to_get = params_to_get
        self.kwargs = kwargs if kwargs else None
        if save_to: 
            self.counter = 0#if path is specified, keep a counter
            self.save_to = save_to 
        if record_input is True:
            fn = partial(self._fn_in_out_params, record_what = 'input') 
        elif record_output is True:
            fn = partial(self._fn_in_out_params, record_what = 'output')  
        elif record_params is True:
            fn = partial(self._fn_in_out_params, record_what = 'params') 
            
        if custom_fn is not None: 
            fn = self._custom_wrapper
            self.custom_fn = custom_fn
            
        if backward is False:
            self.hook = module.register_forward_hook(fn)
        elif backward is True:
            self.hook = module.register_full_backward_hook(fn)
            
    def _fn_in_out_params(self, module, input, output, record_what = None):
        att = getattr(self, 'save_to', None)
        if att is None:
            if record_what == 'input': 
                self.recording = input
            elif record_what == 'output':
                self.recording = output
            elif record_what == 'params':
                params = get_model_params(module, params_to_get = self.params_to_get)[0]
                self.recording = params 
        else:
            name = 'recording_' + str(self.counter) 
            filename = Path(self.save_to) / name
            self.counter += 1
            with open(filename, 'wb') as handle:
                if record_what == 'input': 
                    pickle.dump(input, handle, protocol = pickle.HIGHEST_PROTOCOL)
                elif record_what == 'output':
                    pickle.dump(output, handle, protocol = pickle.HIGHEST_PROTOCOL)
                elif record_what == 'params':
                    params = get_model_params(module, params_to_get = self.params_to_get)[0]
                    pickle.dump(params, handle, protocol = pickle.HIGHEST_PROTOCOL)
                
    def _custom_wrapper(self, module, input, output):
        if self.kwargs: 
            res = self.custom_fn(module, input, output, **self.kwargs)
        else:
            res = self.custom_fn(module, input, output)
        att = getattr(self, 'save_to', None)
        if res and att is None:    
            self.recording = res
        elif res and att:
            name = 'recording_' + str(self.counter) 
            filename = Path(self.save_to) / name
            self.counter += 1
            with open(filename, 'wb') as handle:
                pickle.dump(res, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
    def close(self):
        self.hook.remove()
        att = getattr(self, 'counter', None)
        if att: self.counter = 0
        
def get_all_layers(model):
    '''
    Get all the children (layers) from a model, even the ones that are nested
    
    Input
    -----
    model: class instance based on the base class torch.nn.Module
        
    Output
    ------
    all_layers: list of all layers of the model
    
    Adapted from:
    https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    '''
    children = list(model.children())
    all_layers = []
    if not children:#if model has no children model is last child
        return model
    else:
       # Look for children from children to the last child
       for child in children:
            try:
                all_layers.extend(get_all_layers(child))
            except TypeError:
                all_layers.append(get_all_layers(child))
            
    return all_layers