#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:26:51 2021

@author: yoshi
"""

from gym.envs.registration import register

register(
    id='basic-v0',
    entry_point='spacecraft.basic_env1:SpaceCraftEnv',
)
