#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:34:52 2021

@author: yoshi
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
# from TBot_Functions import *

from pyquaternion import Quaternion

from TBot_Functions import *
# st = dt.datetime.strptime('2020-07-30 11:50:00', '%Y-%m-%d %H:%M:%S')
# st.isoformat()

class SpaceCraftEnv(gym.Env):
    """A spacecraft navigating the solar system environment built for OpenAI gym"""
    def __init__(self):
        super(SpaceCraftEnv,self).__init__()

        #observations
        #keep track of current time
        self.time = astropy.time.Time(START_TIME,format='isot',scale='utc')
        #define timespan for observations
        self.startTime = self.time
        self.tspan = 3600 #project 1 hour into the future
        self.endTime = self.time+dt.timedelta(seconds=self.tspan) 
        self.stepTime = STEP_TIME
        #generates 3601 entries for each planet
        self.ed = generateEphDict(self.startTime,self.endTime,self.tspan)
        #keep track of current step
        self.stepNum = 0
        #keep track of origin position and velocity
        self.r_origin = self.ed[ORIGIN].r[0][0:3]
        self.v_origin = self.ed[ORIGIN].v[0][0:3]
        self.pp = None
        self.vp = None

        
        #spacecraft physical parameters
        self.position = self.r_origin+INIT_POS #relative to sun
        self.velocity = self.v_origin+INIT_VEL #relative to sun
        
       # %control related parameters
        self.orientation = INIT_ORIENT #relative to sun
        self.fuelMass = FUEL_LEVEL_INITIAL
        self.mass = self.fuelMass+SC_MASS_EMPTY
        self.maxThrust = MAIN_ENGINE_THRUST
        self.throttle = 0
        self.action = None
        
        #solution parameters to track
        self.soltol = SOL_TOL
        self.reward = 0
        self.state = None
        self.done = False
        self.impact = False
        
        #define action space
        # self.action_space = spaces.Discrete(4) #this is a discrete action space
        
        self.action_space = spaces.Box(low=np.array([-1,-1,-1,0]),
                                       high=np.array([1,1,1,1]),
                                       dtype=np.float32
                                       ) #this is a continuous action space
        ## roll,pitch,yaw,throttle
        

        #define observation space
        # continuous observation space with lower and upper bounds in obs[:,0],obs[:,1] respectively
        # self.observation_space = spaces.Box(obs[:,0],obs[:,1],dtype=np.float64)
        obs = defineObsSpace(self)
        self.observation_space = spaces.Box(low=np.array(obs[:,0]),
                                            high=np.array(obs[:,1]),
                                            dtype = np.float32
                                            )
        
    def reset(self):
        self.time = astropy.time.Time(START_TIME,format='isot',scale='utc')
        #define timespan for observations
        self.startTime = self.time
        self.endTime = self.time+dt.timedelta(seconds=self.tspan) 
        #generates 3601 entries for each planet
        self.ed = generateEphDict(self.startTime,self.endTime,self.tspan)
        self.stepNum = 0
        self.position = self.r_origin+INIT_POS #relative to sun
        self.velocity = self.v_origin+INIT_VEL #relative to sun
        self.orientation = INIT_ORIENT #relative to sun
        self.fuelMass = FUEL_LEVEL_INITIAL
        self.mass = self.fuelMass+SC_MASS_EMPTY
        self.throttle = 0
        self.reward = 0
        self.state = None
        self.done = False
        self.impact = False
        self.action = None
        self.state = observeState(self)
        
        
        return np.array(self.state)    
    
    
    def _take_action(self,action):
        
        #what does a delta.orientation look like
        # 3 angles (rotate abt x,y,z )
        roll = Quaternion(axis=[1,0,0], angle = 3.14159265*action[0])
        pitch = Quaternion(axis=[0,1,0], angle = 3.14159265*action[1])
        yaw = Quaternion(axis=[0,0,1], angle = 3.14159265*action[2])
        self.orientation = pitch*roll*yaw.rotate(self.orientation)
        self.fuelMass -= 5*sum(action[0:3])
        #what does a thrust look like
        # throttle 0-100
        self.throttle = action[3]
        
        
        return action
        
    def step(self,action):
        #compute reward for current state
        self.reward = getReward(self,action)
        
        
        #take an action based on last state and reward
        self._take_action(action)
        
        #update ephemeris dictionary if necessary
        if self.time > self.endTime:
            self.endTime = self.time+dt.timedelta(seconds=self.tspan)
            self.stepNum = 0
            self.ed = generateEphDict(self.time,self.endTime,self.tspan)
            
        
        #update to new state
        anet = [0,0,0]
        for planet in BODY_DICT.keys():
            anet = np.vstack((anet, getBodyAccel(self,planet)))
        anet = np.sum(anet,axis=0)
        
        #compute acceleration from thrust added to system in +self.orientation direction
        if self.throttle > 0:
            anet += self.action[3]*(self.maxThrust/self.mass)*self.orientation
            self.fuelMass -= self.throttle*M_DOT
            
        self.mass = self.fuelMass+SC_MASS_EMPTY    
        # increment one time step within the environment
        newTime = self.time+dt.timedelta(seconds=self.stepTime)
        #compute kinematic equations of motion for 1 time step and update spacecraft position,velocity
        self.position,self.velocity = spacecraftEOM(self,anet,self.stepTime)
        
        # if self.
        #update observed state
        self.state = observeState(self)
        #update SC time
        self.time = newTime
        #check if done
        self.done = checkDone(self)
        #increment one step
        self.stepNum += 1
        
        return self.state,self.reward,self.done,{}
    

    
    def render(self,mode='human'):
        print('render')


# class SpaceCraftEnv(gym.Env):
#     """A spacecraft navigating the solar system environment for OpenAI gym"""
#     # metadata = {'render.modes': ['human']}
#     def __init__(self):
#         super(SpaceCraftEnv, self).__init__()
#         # Actions of the format +x_thrust in local frame, rotate about x y z +/- 1 degree in local frame, & hold
#         # self.action_space = gym.spaces.Box(1,8, shape=(8,1),dtype=np.float16)
#         self.action_space = spaces.Discrete(4)
#         # self.action_space = spaces.Box()
#         # 39 observations: x,y,z for spacecraft and each planet (30), vx,vy,vz for spacecraft (3),
#         # fuel level for spacecraft (1), and orientation for spacecraft(3), spacecraft throttle and current action
#         obs = np.full((40,2),float('inf'))
#         #planet and SC position + velocity
#         for x in range(0,33):
#             obs[x][0] = float('-inf')
#         #SC orientation is a 1x3 unit vector
#         for x in range(33,36):
#             obs[x][0] = 0
#             obs[x][1] = 1
#             # print('two')
#             # print(obs)
#         #fuel level
#         obs[36][0] = 0
#         obs[36][1] = FUEL_LEVEL_INITIAL
#         #throttle
#         obs[37][0] = False
#         obs[37][1] = True
#         #action
#         obs[38][0] = 0
#         obs[38][1] = 7
#         #time
#         obs[39][0] = convertTimeToJulian(START_TIME)
#         obs[39][1] = float('inf')
#         # print('three')
#         # print(obs)
        
#         self.observation_space = spaces.Box(obs[:,0],obs[:,1],dtype=np.float64)
#         self.total_action_count = 1
#         self.tolerance = TOLERANCE #tolerance for solution
#         self.stepTime = STEP_TIME #seconds
#         self.time = START_TIME
#         self.elapsedTime = 0
#         self.fuel_level = FUEL_LEVEL_INITIAL
#         self.throttle = 0
#         self.mass = self.fuel_level + SC_MASS_EMPTY  # spacecraft mass is 2000 kg plus 10k kg fuel
#         # circular orbit about earth: initial conditions
#         r_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time,self.time+dt.timedelta(seconds=1))[0][0:3]
#         v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time,self.time+dt.timedelta(seconds=1))[1][0:3]
#         # r_earth, v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)
#         self.position = r_earth + INIT_POS  # position relative to Sun
#         self.velocity = v_earth + INIT_VEL  # velocity relative to Sun
#         self.startPos = r_earth + INIT_POS
#         self.startVel = v_earth + INIT_VEL
#         self.action = 1
#         self.orientation = INIT_ORIENT  # orientation relative to central body frame
#         self.target = TARGET_PLANET
#         self.orbit = TARGET_ORBIT
#         self.reward = 0  # initialize reward
#         self.state = None
#         self.done = False
#         self.prevPos = self.position
#         self.beginxfer = [False,False]
#         self.lockEllipse = False
#         self.tracker = 0
#         self.impact = False
#         self.x = []
#         self.y = []
#         self.z = []


    
#     def _take_action(self, action):
#         action_type = action
#         self.total_action_count += 1
#         if action_type == 1:
#             self.throttle = 1
#         elif action_type == 2:
#             # self.orientation = [0, -1, 0]
#             alpha = 10*180/np.pi
#             self.throttle = 0
#             self.fuel_level -= 1
#             self.orientation = self.orientation @ RotationMatrix(alpha, 0, 0)
#         elif action_type == 3:
#             # self.orientation = [0, 1, 0]
#             alpha = -10*180/np.pi
#             self.throttle = 0
#             self.fuel_level -= 1
#             self.orientation = self.orientation @ RotationMatrix(alpha, 0, 0)
#         elif action_type == 4:
#             beta = 10*180/np.pi
#             self.throttle = 0
#             self.fuel_level -= 1
#             self.orientation = self.orientation @  RotationMatrix(0, beta, 0)
#         elif action_type == 5:
#             beta = -10*180/np.pi
#             self.throttle = 0
#             self.fuel_level -= 1
#             self.orientation = self.orientation @  RotationMatrix(0, beta, 0)
#         elif action_type == 6:
#             gamma = 10*180/np.pi
#             self.throttle = 0
#             self.fuel_level -= 1
#             self.orientation = self.orientation @  RotationMatrix(0, 0, gamma)
#         elif action_type == 7:
#             gamma = -10*180/np.pi
#             self.throttle = 0
#             self.fuel_level -= 1
#             self.orientation = self.orientation @ RotationMatrix(0, 0, gamma)
#         elif action_type == 0:
#             # do nothing, coast
#             self.throttle = 0     
#             self.fuel_level -= 1
#             # self.fuel_level-= 1
        
    
#     def step(self, action):
#         # compute reward for current state
#         self.reward = getReward(self,TARGET_PLANET,TARGET_ORBIT,action)
        
#         #track the elapsed time
#         #take an action based on last state and reward
#         self._take_action(action)
        
#         #print('action '+str(action))
#         # print('velocity '+str(np.linalg.norm(self.velocity)))
#         #print('reward ' +str(self.reward))
#         print('fuel level '+str(self.fuel_level))
#         # print('orientation' + str(self.orientation))
#        # print('distance to target ' +str(abs(np.linalg.norm(np.subtract(self.position,planetPositionVelocity(BODY_DICT[TARGET_PLANET].id,self.time)[0]) - TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius))/1e3) + ' Mm')


#         #update to new state
#         Fnet = [0,0,0]
#         #compute net force on space craft from bodies
#         # Fnet = np.sum(getBodyForce(self,planet,self.time)
#         #                for planet in BODY_DICT.keys())
#         for planet in BODY_DICT.keys():
#             Fnet = np.vstack((Fnet, getBodyForce(self,planet,self.time)))
#         Fnet = np.sum(Fnet,axis=0)
#         print('Fnet' + str(Fnet))
        
#         #if throttle is on (ie engine is firing) and fuel remains in the tank then add thrust to Fnet in direction of orientation
#         if self.throttle == 1 and self.fuel_level > 0:
#             self.fuel_level -= M_DOT * self.stepTime #multiply mass flow rate (kg/s) by step-time of fuel burned
#             self.mass -= self.fuel_level #reduce spacecraft mass by fuel burned
#             # update the orientation of SC as a unit vector
#             scDirection = self.orientation / np.linalg.norm(self.orientation) #[0 1 0] / 1
#             Fnet += 0 #set to 0 for testing
#             #Fnet += MAIN_ENGINE_THRUST * scDirection  # add thrust to Fnet in the +direction of orientation

#         # increment one time step within the environment
#         newTime = self.time+dt.timedelta(minutes=self.stepTime/60)
#         #compute kinematic equations of motion for 1 time step
#         newState = spacecraftEOM(self,Fnet,(newTime-self.time))
       
#         #update spacecraft state
#         self.prevPos = self.position
#         self.position = newState[0]
#         self.velocity = newState[1]
#         # update the observed spacecraft state
#         self.state = observeState(self)

#         self.time = newTime
#         self.done = checkDone(self)
        
#         return self.state, self.reward, self.done, {}
            
#     def reset(self):
#         # reset the environment state to an initial state, in orbit about Earth
#         self.time = START_TIME
#         self.elapsedTime = 0
#         self.fuel_level = FUEL_LEVEL_INITIAL
#         self.throttle = 0
#         self.mass = FUEL_LEVEL_INITIAL + 2000  # spacecraft mass is 2000 kg plus 10k kg fuel
#         # circular orbit about earth: initial conditions
#         r_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)[0]
#         v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1]
#         self.position = self.startPos  # position relative to Sun
#         self.velocity = self.startVel  # velocity relative to Sun
#         self.orientation = np.array([0, 1, 0])  # orientation relative to central body frame
#         self.done = False
#         self.reward = 0  # initialize reward
#         self.state = observeState(self)
#         self.beginxfer = [False,False]
#         self.lockEllipse = False
#         self.tracker = 0
#         self.impact = False
#         self.x = []
#         self.y = []
#         self.z = []
        
        
#         return np.array(self.state)
        
#     def render(self, mode='human'):
#         print('render this')
 


