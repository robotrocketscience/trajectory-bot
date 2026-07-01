#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:15:41 2021

@author: robotrocketscience
"""


from TBot_Setup import *


############### INPUTS ######################
MF = 7.69*2 #double the fuel for circularization on other side
SC_MASS_EMPTY = 50
SC_MASS_FULL = SC_MASS_EMPTY*MF
FUEL_LEVEL_INITIAL = (SC_MASS_FULL - SC_MASS_EMPTY)# kg
# assume g0 = 9.81, Isp = 400 seconds
# mdot = F/(Isp*g0)
MAIN_ENGINE_THRUST = 1 # kiloNewtons
M_DOT = MAIN_ENGINE_THRUST/(400*9.81) # kg/s, mass flow rate of engine

angle = random.uniform(0,2*math.pi)
INIT_POS = np.array([BODY_DICT['earth'].radius+150,0,0])  # x y z, relative to earth, need to convert to be relative to sun
INIT_VEL = np.array([0, math.sqrt(BODY_DICT['earth'].mu/np.linalg.norm(INIT_POS)), 0])  # vx vy vz relative to earth, need to convert to be relative to sun
INIT_ORIENT = np.array([0,1,0]) #relative to central body frame

START_TIME = '2020-07-30T11:50:00.0'
# START_TIME = dt.datetime.strptime('2020-07-30 11:50:00', '%Y-%m-%d %H:%M:%S')

STEP_TIME = 1 #seconds
SOL_TOL = 0.001 #solution tolerance

ORIGIN = 'earth'
TARGET_PLANET = 'moon' #the target object to orbit
TARGET_ORBIT = 50 #the target circular orbital altitude
