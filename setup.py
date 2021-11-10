#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

DESCRIPTION = 'collection of PyTorch utilities to accomplish tasks relevant for many projects without the need to re-write the same few lines of code again and again'
with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='torchknickknacks',
      version='0.1.2',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url='https://github.com/AlGoulas/torchknickknacks',
      author='Alexandros Goulas',
      author_email='agoulas227@gmail.com',
      python_requires='>=3.7',
      packages=find_packages(),
      include_package_data=True,
      classifiers=[      
          "Programming Language :: Python :: 3.9",
          "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)