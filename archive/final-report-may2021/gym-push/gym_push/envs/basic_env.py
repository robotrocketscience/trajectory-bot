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
import numpy as np
import datetime as dt
from scipy import integrate
from collections import namedtuple
import math
import random
#NOTES
#deltaV required for LEO to LLO is approximately 8 km/s
# therefore mass fraction of spacecraft should be 7.69
# EXPERIMENTAL CONDITIONS
MF = 7.69
SC_MASS_EMPTY = 50
SC_MASS_FULL = SC_MASS_EMPTY*MF
FUEL_LEVEL_INITIAL = 2*(SC_MASS_FULL - SC_MASS_EMPTY)# kg
# assume g0 = 9.81, Isp = 400 seconds
# mdot = F/(Isp*g0)
M_DOT = 0.2548  # kg/s, mass flow rate of engine
MAIN_ENGINE_THRUST = .1000  # kNewtons
angle = random.uniform(0,2*math.pi)
INIT_POS = np.array([6528*math.cos(angle),6528*math.sin(angle), 0])  # x y z, relative to earth, need to convert to be relative to sun
INIT_VEL = np.array([0, 7.81409, 0])  # vx vy vz relative to earth, need to convert to be relative to sun
INIT_ORIENT = np.array([0,1,0]) #relative to central body frame
START_TIME = dt.datetime.strptime('2020-07-30 11:50:00', '%Y-%m-%d %H:%M:%S') #Mars 2020 launch date, 
# converted to Julian date
STEP_TIME = 60
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

def getBodyForce(self,planet,time):
        mu = BODY_DICT[planet].mu
        r_body = planetPositionVelocity(BODY_DICT[planet].id, time)[0]
        return mu*(r_body-self.position)/np.linalg.norm(r_body-self.position) **3
    
def checkDone(self):
    #define distance from body to sc for easier book keeping
    rvec = self.position - planetPositionVelocity(BODY_DICT[self.target].id,self.time)[0]
    # define circular velocity of SC about target for easier bookkeeping
    vvec = self.velocity - planetPositionVelocity(BODY_DICT[self.target].id,self.time)[1]
    targetV = math.sqrt(BODY_DICT[TARGET_PLANET].mu/(TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius))
    targetT = 2*math.pi*math.sqrt((TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius)**3 /BODY_DICT[TARGET_PLANET].mu)
    targetE = -BODY_DICT[TARGET_PLANET].mu/(2*TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius)
        # compute these values for the spacecraft, relative to the target
    sc_vel1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[0]
    sc_T1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[1]
    sc_E1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[2]
        # no fuel left is DONE
    if self.fuel_level <= 0:
        done = True   
        print('out of fuel')
    #achieve the desired orbit is DONE
    elif (abs(targetV-sc_vel1) <= self.tolerance) and (abs(targetT-sc_T) <= self.tolerance) and (abs(targetE-sc_E1) <= self.tolerance):
        done = True
        print('orbit achieved')
    # elif self.reward < -1e15:
        # done = True
    else:    
        done = False
    #crash into a body is DONE, come back to this condition later
    # for planet in BODY_DICT.keys():
    #     if np.linalg.norm(planetPositionVelocity(BODY_DICT[planet].id,self.time)[0]) <= np.linalg.norm(self.position): 
    #         done = True
    #         break
    return done
    
# def isOvershot(self,targetPlanet):
#     originOfPlane = planetPositionVelocity(BODY_DICT[targetPlanet].id, self.time)[0]
#     p1 = np.array([originOfPlane[0] + BODY_DICT[targetPlanet].radius/100, originOfPlane[1], originOfPlane[2]])
#     p2 = np.array([originOfPlane[0] , originOfPlane[1]+ BODY_DICT[targetPlanet].radius/100, originOfPlane[2]])
#     p3 = np.array([originOfPlane[0] , originOfPlane[1], originOfPlane[2]])
#     v1 = p3-p1
#     v2 = p2-p1
#     cp = np.cross(v1,v2)
#     a,b,c = cp
#     d = np.dot(cp,p3) #d is a plane
#     rmag = np.linalg.norm(self.position-planetPositionVelocity(BODY_DICT[targetPlanet].id, self.time)[0])
#     # self.orientation [0 1 0]
#     #rotate plane towards spacecraft
#     #check orthogonality of norm vs self.orientiation
#     # if not orthogonal, fail
#     # if yes orthogonal, gradient reward as approaches parallel
#     #or do a cone
    
# def buildOptPath(self):
#     #construct an ellipse whos foci are the origin and target body at each time step
#     F1 = planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0]
#     F2 = planetPositionVelocity(BODY_DICT['moon'].id,self.time)[0]
#     #semi major axis is the distance between the bodies, plus each body radius, plus the orbit altitude
#     sma = np.linalg.norm(F1-F2) + BODY_DICT['earth'].radius + BODY_DICT['moon'].radius + TARGET_ORBIT + 150
                                          
  
    
     
    
def spacecraftEOM(self,Fnet,time):
    time = time.total_seconds()
    v = self.velocity + (Fnet/self.mass)*time
    r = self.position + self.velocity*time + 0.5*(Fnet/self.mass)*time**2
    # print('Fnet '+str(np.linalg.norm(Fnet)))
    return r,v
        
def observeState(self):
    #organize planet position vectors in one column, descending order from sun to neptune
    planet_positions = np.hstack([planetPositionVelocity(BODY_DICT[planet].id, self.time)[0] for planet in BODY_DICT.keys()]).T
    temp = np.delete(planet_positions,[0,1,2]) #delete sun entries as they are 0
    temp = np.append(temp,self.position) #append SC position to the state
    temp = np.append(temp,self.velocity) #append SC velocity to the state
    temp = np.append(temp,self.orientation) #append orientation
    temp = np.append(temp,self.fuel_level) #append fuel level
    temp = np.append(temp,self.throttle) #append throttle (true/false)
    temp = np.append(temp,self.action)
    temp = np.append(temp,convertTimeToJulian(self.time))
    return temp

def getReward(self,target,desired_orbit,action):
        #a circular orbit is defined by:
        #speed v = sqrt(mu/r) for all t
        # period T = 2*pi*sqrt(r^3 /mu)
        #orbit energy E = -mu/(2*r)
        #compute the target 
    targetV = math.sqrt(BODY_DICT[target].mu/(TARGET_ORBIT+BODY_DICT[target].radius))
    targetT = 2*math.pi*math.sqrt((TARGET_ORBIT+BODY_DICT[target].radius)**3 /BODY_DICT[target].mu)
    targetE = -BODY_DICT[target].mu/(2*TARGET_ORBIT+BODY_DICT[target].radius)
    targetVec = planetPositionVelocity(BODY_DICT[TARGET_PLANET].id, self.time)[0] #distance from sun to moon
        # compute these values for the spacecraft, relative to the target
    sc_vel1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[0]
    sc_T1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[1]
    sc_E1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[2]
    r_sc_m = targetVec - self.position #vector distance from spacecraft to moon

    reward = 0
    # delr = self.position - planetPositionVelocity(BODY_DICT[target].id,self.time)[0]
    # delv = self.velocity - planetPositionVelocity(BODY_DICT[target].id,self.time)[1]
    # k = math.sqrt(delr[0]**2 + delr[1]**2 +delr[2]**2 + delv[0]**2 + delv[1]**2 +delv[2]**2)
    # scaleFactor = 1e-5 
    # psi = FUEL_LEVEL_INITIAL
    # if np.linalg.norm(delr) < np.linalg.norm(planetPositionVelocity(BODY_DICT[target].id,self.time)[0]) and np.linalg.norm(delv) < np.linalg.norm(planetPositionVelocity(BODY_DICT[target].id,self.time)[1]):
    #     reward = .5*psi/(1+self.total_action_count)*math.exp(-scaleFactor*k)
    # else:
    #     reward = -10
    # print('del_reward: ' + str(reward))
        
    #orbital parameter achievement awards
    reward += (1-(sc_vel1/targetV)**.2)
    reward += .1*(1-(sc_T1/targetT)**.2)
    reward += (1-(sc_E1/targetE)**.2)
    #arrival condition met bonus
    if abs(sc_vel1-targetV) <= self.tolerance:
        reward += 100
    if abs(sc_T1-targetT) <= self.tolerance:
       reward += 100
    if abs(sc_E1 - targetE) <= self.tolerance:
        reward += 100
        
    #distance reward
    reward += .1*(1-(np.linalg.norm(self.position-planetPositionVelocity(BODY_DICT[target].id,self.time)[0])+BODY_DICT[target].radius)**.2)
    
    #impact penalty
    if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT[target].id,self.time)[0])) <= BODY_DICT[target].radius:
        reward -= -1000
    if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0])) <= BODY_DICT['earth'].radius:
        reward -= -1000
        
    #punish changing orientation so damn much
    if action > 1:
        reward -= 2
    # if not np.array_equal(self.orientation,INIT_ORIENT):
    #     reward -= 5*(abs(self.orientation[0]-INIT_ORIENT[0]) + abs(self.orientation[1]-INIT_ORIENT[1]) + abs(self.orientation[2]-INIT_ORIENT[2]))
    
    #punish using fuel
    fuel_expended = FUEL_LEVEL_INITIAL - self.fuel_level
    reward -= 10*math.log10(fuel_expended+1)
    if self.fuel_level <= 0:
        reward -= 100
    
    # prevTargDistance = np.linalg.norm(self.prevPos-planetPositionVelocity(BODY_DICT[target].id, self.time-dt.timedelta(minutes=1))[0])
    # currTargDistance = np.linalg.norm(self.position-planetPositionVelocity(BODY_DICT[target].id, self.time)[0])
    # if currTargDistance >= prevTargDistance:
    #     reward += 10
    #     if action == 0:
    #         reward -= 10
            
    # reward increasing velocity to moon orbit vel and closing distance to moon
    # velocity_reward = 10*(1 - (sc_vel1/targetV)**0.2)
    # distance_reward = 10*(math.log10(np.linalg.norm(r_sc_m)/BODY_DICT[target].radius+desired_orbit))
    # reward += velocity_reward + distance_reward
    # print('earth-relative vel ' +str(np.linalg.norm(self.velocity - planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1])))
    # if np.linalg.norm(self.velocity - planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1]) >= 7.81409+3.02:
    #     reward -= 10*math.log10(fuel_expended+1)
    #maybe add earth escape velocity as a reward condition
    #if no more thrust is needed, (ie reaches correct speed) punish heavily for thrusting
    # self.velocity > 
    
    #punish lowering speed wrt earth, punish lowering orbit altitude wrt earth
    # if np.lingalg.norm(self.velocity)
    
    return reward 
   

def getOrbitParams(state,time,target):
    r_m_sc = state - planetPositionVelocity(BODY_DICT[target].id,time)[0] #position vector of spacecraft relative to moon
    # v_m_sc = self.velocity - planetPositionVelocity(BODY_DICT[target].id,self.time)[1] #velocity vector of spacecraft relative to moon
        
    currentV = math.sqrt(BODY_DICT[target].mu/(np.linalg.norm(r_m_sc)+BODY_DICT[target].radius))
    currentT = 2*math.pi*math.sqrt(((np.linalg.norm(r_m_sc)+BODY_DICT[target].radius)**3 /BODY_DICT[target].mu))
    currentE = -BODY_DICT[target].mu / (2 * (np.linalg.norm(r_m_sc) + BODY_DICT[target].radius))
    return currentV,currentT,currentE,r_m_sc


class SpaceCraftEnv(gym.Env):
    """A spacecraft navigating the solar system environment for OpenAI gym"""
    
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        super(SpaceCraftEnv, self).__init__()
        # Actions of the format +x_thrust in local frame, rotate about x y z +/- 1 degree in local frame, & hold
        # self.action_space = gym.spaces.Box(1,8, shape=(8,1),dtype=np.float16)
        self.action_space = spaces.Discrete(4)

        # 39 observations: x,y,z for spacecraft and each planet (30), vx,vy,vz for spacecraft (3),
        # fuel level for spacecraft (1), and orientation for spacecraft(3), spacecraft throttle and current action
        obs = np.full((40,2),float('inf'))
        #planet and SC position + velocity
        for x in range(0,33):
            obs[x][0] = float('-inf')
        #SC orientation is a 1x3 unit vector
        for x in range(33,36):
            obs[x][0] = 0
            obs[x][1] = 1
            # print('two')
            # print(obs)
        #fuel level
        obs[36][0] = 0
        obs[36][1] = FUEL_LEVEL_INITIAL
        #throttle
        obs[37][0] = False
        obs[37][1] = True
        #action
        obs[38][0] = 0
        obs[38][1] = 7
        #time
        obs[39][0] = convertTimeToJulian(START_TIME)
        obs[39][1] = float('inf')
        # print('three')
        # print(obs)
        
        self.observation_space = spaces.Box(obs[:,0],obs[:,1],dtype=np.float64)
        self.total_action_count = 1
        self.tolerance = TOLERANCE #tolerance for solution
        self.stepTime = STEP_TIME #seconds
        self.time = START_TIME
        self.elapsedTime = 0
        self.fuel_level = FUEL_LEVEL_INITIAL
        self.throttle = 0
        self.mass = self.fuel_level + SC_MASS_EMPTY  # spacecraft mass is 2000 kg plus 10k kg fuel
        # circular orbit about earth: initial conditions
        r_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)[0]
        v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1]
        # r_earth, v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)
        self.position = r_earth + INIT_POS  # position relative to Sun
        self.velocity = v_earth + INIT_VEL  # velocity relative to Sun
        self.startPos = r_earth + INIT_POS
        self.startVel = v_earth + INIT_VEL
        self.action = 1
        self.orientation = INIT_ORIENT  # orientation relative to central body frame
        self.target = TARGET_PLANET
        self.orbit = TARGET_ORBIT
        self.reward = 0  # initialize reward
        self.state = None
        self.done = False
        self.prevPos = self.position


    
    def _take_action(self, action):
        action_type = action
        self.total_action_count += 1
        if action_type == 1:
            self.throttle = 1
        elif action_type == 2:
            # self.orientation = [0, -1, 0]
            alpha = 10*180/np.pi
            self.throttle = 0
            self.fuel_level -= 5
            self.orientation = self.orientation @ RotationMatrix(alpha, 0, 0)
        elif action_type == 3:
            # self.orientation = [0, 1, 0]
            alpha = -10*180/np.pi
            self.throttle = 0
            self.fuel_level -= 5
            self.orientation = self.orientation @ RotationMatrix(alpha, 0, 0)
        elif action_type == 4:
            beta = 10*180/np.pi
            self.throttle = 0
            self.fuel_level -= 10
            self.orientation = self.orientation @  RotationMatrix(0, beta, 0)
        elif action_type == 5:
            beta = -10*180/np.pi
            self.throttle = 0
            self.fuel_level -= 10
            self.orientation = self.orientation @  RotationMatrix(0, beta, 0)
        elif action_type == 6:
            gamma = 10*180/np.pi
            self.throttle = 0
            self.fuel_level -= 10
            self.orientation = self.orientation @  RotationMatrix(0, 0, gamma)
        elif action_type == 7:
            gamma = -10*180/np.pi
            self.throttle = 0
            self.fuel_level -= 10
            self.orientation = self.orientation @ RotationMatrix(0, 0, gamma)
        elif action_type == 0:
            # do nothing, coast
            self.throttle = 0     
            # self.fuel_level-= 1
        
    
    def step(self, action):
        # compute reward for current state
        self.reward = getReward(self,TARGET_PLANET,TARGET_ORBIT,action)
        
        #track the elapsed time
        #take an action based on last state and reward
        self._take_action(action)
        
        print('action '+str(action))
        # print('velocity '+str(np.linalg.norm(self.velocity)))
        print('reward ' +str(self.reward))
        print('fuel level '+str(self.fuel_level))
        # print('orientation' + str(self.orientation))
        print('distance to target ' +str(abs(np.linalg.norm(np.subtract(self.position,planetPositionVelocity(BODY_DICT[TARGET_PLANET].id,self.time)[0]) - TARGET_ORBIT+BODY_DICT[TARGET_PLANET][2]))/1e3) + ' kme3')


        #update to new state
        Fnet = [0,0,0]
        #compute net force on space craft from bodies
        # Fnet = np.sum(getBodyForce(self,planet,self.time)
        #                for planet in BODY_DICT.keys())
        for planet in BODY_DICT.keys():
            Fnet = np.vstack((Fnet, getBodyForce(self,planet,self.time)))
        Fnet = np.sum(Fnet,axis=0)
        
        #if throttle is on (ie engine is firing) and fuel remains in the tank then add thrust to Fnet in direction of orientation
        if self.throttle == 1 and self.fuel_level > 0:
            self.fuel_level -= M_DOT * self.stepTime #multiply mass flow rate (kg/s) by step-time of fuel burned
            self.mass -= self.fuel_level #reduce spacecraft mass by fuel burned
            # update the orientation of SC as a unit vector
            scDirection = self.orientation / np.linalg.norm(self.orientation) #[0 1 0] / 1
            Fnet += MAIN_ENGINE_THRUST * scDirection  # add thrust to Fnet in the +direction of orientation

        # increment one time step within the environment
        newTime = self.time+dt.timedelta(minutes=self.stepTime/60)
        #compute kinematic equations of motion for 1 time step
        newState = spacecraftEOM(self,Fnet,(newTime-self.time))
       
        #update spacecraft state
        self.prevPos = self.position
        self.position = newState[0]
        self.velocity = newState[1]
        # update the observed spacecraft state
        self.state = observeState(self)

        self.time = newTime
        self.done = checkDone(self)
        
        return self.state, self.reward, self.done, {}
            
    def reset(self):
        # reset the environment state to an initial state, in orbit about Earth
        self.time = START_TIME
        self.elapsedTime = 0
        self.fuel_level = FUEL_LEVEL_INITIAL
        self.throttle = 0
        self.mass = FUEL_LEVEL_INITIAL + 2000  # spacecraft mass is 2000 kg plus 10k kg fuel
        # circular orbit about earth: initial conditions
        r_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)[0]
        v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1]
        self.position = r_earth + INIT_POS  # position relative to Sun
        self.velocity = v_earth + INIT_VEL  # velocity relative to Sun
        self.orientation = np.array([0, 1, 0])  # orientation relative to central body frame
        self.done = False
        self.reward = 0  # initialize reward
        self.state = observeState(self)
        
        
        return np.array(self.state)
        
    def render(self, mode='human'):
        print('render this')
                
def RotationMatrix(a, b, c):
    return np.array([[np.cos(a) * np.cos(b),
                      np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c),
                      np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)],
                     [np.sin(a) * np.cos(b),
                      np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
                      np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)],
                     [-np.sin(b),
                      np.cos(b) * np.sin(c),
                      np.cos(b) * np.cos(c)]])

def convertTimeToJulian(dtObj):
    q=dateutil.parser.parse(dt.datetime.strftime(dtObj,'%Y-%m-%d %H:%M:%S'))
    time = astropy.time.Time(q)
    return time.jd


def planetPositionVelocity(planet, date_time):
    obj = Horizons(id_type='id',id=planet,location='@sun',epochs=convertTimeToJulian(date_time))
    vec = Table(obj.vectors())
    r = vec.as_array(keep_byteorder=False,names='x,y,z').data
    v = vec.as_array(keep_byteorder=False,names='vx, vy, vz').data
    r.dtype=float
    v.dtype=float
    return r * 1.496e8, v[3:6] * 1.731e3 #convert from AU to km, au/day to km/s



