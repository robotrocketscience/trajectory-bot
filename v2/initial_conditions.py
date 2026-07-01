#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:20:21 2021

@author: robotrocketscience
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from astropy.table import Table
import astropy.time
import dateutil.parser
from astroquery.jplhorizons import Horizons

import datetime as dt
from collections import namedtuple

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import math
import numpy as np


#NOTES
#deltaV required for LEO to LLO is approximately 8 km/s
# therefore mass fraction of spacecraft should be 7.69
# EXPERIMENTAL CONDITIONS

MF = 7.69*2 #double the fuel for circularization on other side
SC_MASS_EMPTY = 50
SC_MASS_FULL = SC_MASS_EMPTY*MF
FUEL_LEVEL_INITIAL = (SC_MASS_FULL - SC_MASS_EMPTY)# kg
# assume g0 = 9.81, Isp = 400 seconds
# mdot = F/(Isp*g0)
MAIN_ENGINE_THRUST = 10 # Newtons
M_DOT = MAIN_ENGINE_THRUST/(400*9.81) # kg/s, mass flow rate of engine

angle = random.uniform(0,2*math.pi)
INIT_POS = np.array([6528,0,0])  # x y z, relative to earth, need to convert to be relative to sun
INIT_VEL = np.array([0, math.sqrt(398600/6528), 0])  # vx vy vz relative to earth, need to convert to be relative to sun
INIT_ORIENT = np.array([0,1,0]) #relative to central body frame
START_TIME = dt.datetime.strptime('2020-07-30 11:50:00', '%Y-%m-%d %H:%M:%S') #Mars 2020 launch date, 
# converted to Julian date
STEP_TIME = 1 #seconds
TOLERANCE = 0.001
TARGET_PLANET = 'moon' #the target object to orbit
TARGET_ORBIT = 50 #the target circular orbital altitude
#expect to arrive in 266d 16h 59m


MassiveBody = namedtuple(typename='massive_body',
                         field_names=['mu', 'id','radius'])

BODY_DICT = {'sun': MassiveBody(mu=132700000000, id='10',radius=696500),
             'mercury': MassiveBody(mu=2.2032e4, id='1',radius=2441),
             'venus':  MassiveBody(mu=3.2485e5, id='299',radius=6051.89),
             'earth': MassiveBody(mu=398600, id='399',radius=6378),
             'moon': MassiveBody(mu=4902.8 ,id='301', radius=1738),
             'mars': MassiveBody(mu=42820, id='499',radius=3396),
             'jupiter': MassiveBody(mu=1.26686e8, id='599',radius=69911),
             'saturn': MassiveBody(mu=3.79311e7, id='699',radius=58232),
             'uranus': MassiveBody(mu=5.7939e6, id='799',radius=25362),
             'neptune': MassiveBody(mu=6.8365e6, id='899',radius=24624)
             }
