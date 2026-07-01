#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:16:11 2021

@author: robotrocketscience
"""

import numpy as np
import math

pi = math.pi
mu_e =  398600
mu_m = 4902.8

rsoi_e = 924000
rsoi_m = 66100
r_em = 384748 #km distance from earth to moon 
r_hman = r_em - rsoi_m #hohmann transfer radius

a_target = r_em
a_hoh = .5*(150+6378+r_hman+1738)


Thoh = 2*pi*math.sqrt((a_hoh**3) /mu_e) #period of the hohmann transfer ellipse
thdot_target = (360/2*pi)*math.sqrt(mu_e/a_target**3) #phase angle btwn earth and moon

phi = 180-.5*Thoh*thdot_target

