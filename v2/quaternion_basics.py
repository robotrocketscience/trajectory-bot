#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:05:30 2021

@author: yoshi
"""
# http://kieranwynn.github.io/pyquaternion/

from pyquaternion import Quaternion

my_quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)

import numpy
numpy.set_printoptions(suppress=True) # Suppress insignificant values for clarity
v = numpy.array([0., 0., 1.]) # Unit vector in the +z direction
v_prime = my_quaternion.rotate(v)
v_prime
# array([ 0., 0., -1.])

q1 = Quaternion(axis=[1, 0, 0], angle=3.14159265) # Rotate 180 about X
q2 = Quaternion(axis=[0, 1, 0], angle=3.14159265 / 2) # Rotate 90 about Y
q3 = q1 * q2 # Composite rotation of q1 then q2 expressed as standard multiplication
v_prime = q3.rotate(v)
v_prime
    # array([ 1., 0., 0.])