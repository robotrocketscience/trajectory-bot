#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:28:26 2021

@author: yoshi
"""

from setuptools import setup, find_packages

setup(name='gym-push',
      packages=find_packages(),
      include_package_data=True,
      package_data={
        '': ['*.csv', '*.npy'],
      },      
      version='0.0.1',
      install_requires=['gym', 'numpy', 'pandas', 'joblib']
)