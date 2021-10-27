#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchknickknacks import metrics

x1 = torch.rand(100,)
x2 = torch.rand(100,)
r = metrics.pearson_coeff(x1, x2)

x = torch.rand(100, 30)
r_pairs = metrics.pearson_coeff_pairs(x)
