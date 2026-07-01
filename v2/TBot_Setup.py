#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:12:13 2021

@author: robotrocketscience
"""

# # #
# This file sets up the constants and inputs for the Trajectory Bot problem


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

MassiveBody = namedtuple(typename='massive_body',
                         field_names=['mu', 'id','radius'])
                                   
BODY_DICT = {'sun': MassiveBody(mu=132712440018, id='10',radius=696500),
             'mercury': MassiveBody(mu=22032, id='1',radius=2441),
             'venus':  MassiveBody(mu=324859, id='299',radius=6051.89),
             'earth': MassiveBody(mu=398600.4418, id='399',radius=6378),
             'moon': MassiveBody(mu=4904.8695 ,id='301', radius=1738),
             'mars': MassiveBody(mu=42828.37, id='499',radius=3396),
             'jupiter': MassiveBody(mu=126686534, id='599',radius=69911),
             'saturn': MassiveBody(mu=37931187, id='699',radius=58232),
             'uranus': MassiveBody(mu=5793939, id='799',radius=25362),
             'neptune': MassiveBody(mu=6383652.9, id='899',radius=24624)
             }