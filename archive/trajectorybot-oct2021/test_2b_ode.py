#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:49:57 2021

@author: robotrocketscience
"""

# Imports
# SciPy's numerical ODE solver
from scipy.integrate import odeint
# The fabulous numpy library:
import numpy as np
import math
import matplotlib.pyplot as plt

pi = math.pi
# Parameters
# G = 3
mu_e = 398600
G = mu_e
x0 = 150+6378 #1
y0 = 0
z0 = 0
vx0 = 0
vy0 = 7.814+4.04
vz0 = 0

# In the following definition of F, s is a state vector whose components represent the following:
#   s[0]: Horizontal or x position
#   s[1]: Vertical or y position
#   s[2]: Horizontal velocity
#   s[3]: Vertical velocity
# In general, F can depend upon time t as well. Although our F is independent of t, we still need to
# indicate that it is a possible variable.

def F(s,t,G): return [s[3],s[4],s[5],
    -G*s[0]/(s[0]**2 + s[1]**2 + s[2]**2)**(3/2),
    -G*s[1]/(s[0]**2 + s[1]**2 + s[2]**2)**(3/2),
    -G*s[2]/(s[0]**2 + s[1]**2 + s[2]**2)**(3/2),
]


T = 2*pi*math.sqrt((384400) ** 3 /mu_e) #period of the hohmann transfer ellipse
# for testing
# t = np.linspace(0,T,100)
# initS = [150+6378,0,0,0,math.sqrt(mu_e/(150+6378)),0]
# solution = odeint(F,initS,t,args=(mu_e,))
# We'll solve the system for the following values of t, namely
# 100 equally spaced values between 0 and 2.
t = np.linspace(0,6*T,100000)

# Solve it!
solution = odeint(F,[x0,y0,z0,vx0,vy0,vz0],t,args=(mu_e,))

x = solution[:,0]
y = solution[:,1]
z = solution[:,2]
plt.plot(x,y, '.');
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.plot3D(x,y,z)

