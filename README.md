<p align="center">
<img src="knickknacks.png" alt="drawing" width="200"/>
</p>

# torchknickknacks

a collection of PyTorch utilities to accomplish tasks relevant for many projects without the need to re-write the same few lines of code again and again

# Motivation

PyTorch offers a broad pallette of functions and classes for open-ended machine learning projects. **Many recurrent tasks, however, involve putting together a few lines of broadly the same PyTorch code**, for instance, deleting layers from a model, freezing parameters, getting the output of a layer etc., either by consulting forums or the PyTorch docs. The motivation for torchknickknacks is to **collect functions to accomplish such recurrent tasks with one package**.

# Installation

Clone the repository, create a virtual environment (e.g., with conda) and install the requirements. Change into the torchknickknacks folder and type:

```pip install .```

# Features 

**modelutils**

* ```get_model_params``` extract the names, corresponding tensors and ```requires_grad``` attribute from a model
* ```freeze_params``` freezes or unfreezes the parameters of a model
* ```delete_layers``` delete specific layers from a model
* ```add_layers``` add layers in a specific position in a model
* ```Recorder```	record the input, output or parameters of a layer/module of a model during forward or backward passes. Supports custom functions for arbitrary manupulation of modules.

**metrics**

* ```pearson_coeff``` computes pearson correlation coefficient between two 1D tensors   
* ```pearson_coeff_pairs``` computes pearson correlation coefficient across the 1st dimension of a 2D tensor 
* ```accuracy``` compute the accuracy for classification tasks
* ```class weights``` calculate class weights for classficiation with imbalanced classes 

# Examples

An example for each feature is given in ```examples```

# Acknowledgements

Functions in this package are motivated by and based on questions and tips in the PyTorch forum

All functions related to model parameters are based on questions and tips [in this thread](https://discuss.pytorch.org/t/access-all-weights-of-a-model/77672)

Recorder class based on [this tutorial](https://www.kaggle.com/sironghuang/understanding-pytorch-hooks/notebook) on hooks in PyTorch
 

