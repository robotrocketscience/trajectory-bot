#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:57:59 2021

@author: robotrocketscience
"""
from setuptools import setup, find_packages

setup(name='TrajectoryBot_V2',
      packages=find_packages(),
      include_package_data=True,
      package_data={
          '': ['*.csv','*.npy'],
          },
      version='0.0.1',
      install_requires=['gym', 'numpy', 'astropy', 'astroquery', 'datetime', 'pyquaternion']
)
