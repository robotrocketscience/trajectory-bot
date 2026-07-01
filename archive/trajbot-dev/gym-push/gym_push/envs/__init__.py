#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:27:25 2021

@author: yoshi
"""
# import gym

# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'basic-v0' in env:
#          print("Remove {} from registry".format(env))
#          del gym.envs.registration.registry.env_specs[env]
         
from gym_push.envs.basic_env1 import SpaceCraftEnv
# import envs
# import basic_env