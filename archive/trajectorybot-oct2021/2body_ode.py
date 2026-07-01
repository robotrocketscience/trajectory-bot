#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:14:09 2021

@author: robotrocketscience
"""

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import  numpy as np
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 
#2 body problem orbit trajectory generator

pi =  math.pi
def  F(s,t,mu):
    #G is the gravitational parameter mu of central body
    #s is the state vector containing xyz position and velocity
    #t is time
    r = np.linalg.norm(s[0:3])
    sdot = 
    for i in range(3):
        sdot.append((-mu/r**3)*s[i])
    sdot[3:6] = s[3:6]

    return sdot



#define integration time, we can estimate it as 1/2 period T of transfer ellipse
#define initial condition, state vector s at the moment phi_sc = phi_target
mu_e = 398600
T = 2*pi*math.sqrt((150+6378) ** 3 /mu_e) #period of the hohmann transfer ellipse
# for testing
t = np.linspace(0,T,100)
initS = [150+6378,0,0,0,math.sqrt(mu_e/(150+6378)),0]
solution = odeint(F,initS,t,args=(mu_e,))

x = solution[:,0:3]
y = solution[:,0:3]
z = solution[:,0:3]
plt.plot(x,y,'.')

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.plot3D(x,y,z)