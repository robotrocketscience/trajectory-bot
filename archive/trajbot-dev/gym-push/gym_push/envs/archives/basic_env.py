#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:27:27 2021

@author: robotrocketscience
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from astropy.table import Table
from astroquery.jplhorizons import Horizons
import numpy as np
import datetime as dt
from scipy import integrate
from collections import namedtuple


# EXPERIMENTAL CONDITIONS
FUEL_LEVEL_INITIAL = 10000  # kg
# assume g0 = 9.81, Isp = 400 seconds
M_DOT = 0.2548  # kg/s, mass flow rate of engine
MAIN_ENGINE_THRUST = 1000  # Newtons
INIT_POS = np.array([150, 0, 0])  # x y z, relative to earth, need to convert to be relative to sun
INIT_VEL = np.array([0, 63.1347, 0])  # vx vy vz relative to earth, need to convert to be relative to sun
START_TIME = dt.datetime.strptime('2020-07-30 11:50:00', '%Y-%m-%d %H:%M:%S') #Mars 2020 launch date
STEP_TIME = 60
TOLERANCE = 0.001
#expect to arrive in 266d 16h 59m




MassiveBody = namedtuple(typename='massive_body',
                         field_names=['mu', 'id','radius'])

BODY_DICT = {'sun': MassiveBody(mu=1327000000000, id='10',radius=696500),
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


def RotationMatrix(self, a, b, c):
    return np.array([[np.cos(a) * np.cos(b),
                      np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c),
                      np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)],
                     [np.sin(a) * np.cos(b),
                      np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
                      np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)],
                     [-np.sin(b),
                      np.cos(b) * np.sin(c),
                      np.cos(b) * np.cos(c)]])


class SpaceCraftEnv(gym.Env):
    """A spacecraft navigating the solar system environment for OpenAI gym"""

    # metadata = {'render.modes': ['human']}
    def __init__(self):
        super(SpaceCraftEnv, self).__init__()
        # Actions of the format +x_thrust in local frame, rotate about x y z +/- 1 degree in local frame, & hold
        # self.action_space = gym.spaces.Box(1,8, shape=(8,1),dtype=np.float16)
        self.action_space = spaces.Discrete(8)
        
        #40 observations: x,y,z for spacecraft and each planet (33), vx,vy,vz for spacecraft (3),
        # fuel level for spacecraft (1), and orientation for spacecraft(3)
        obs = np.full((40,2),float('inf'))
        for x in range(0,35):
            obs[x][0] = float('-inf')
        obs[36][0] = 0
        obs[36][1] = FUEL_LEVEL_INITIAL
        for x in range(37,40):
            obs[x][0] = 0
            obs[x][1] = 359
            
            #need to add reward to observation space
            
        self.observation_space = spaces.Box(obs[:,0],obs[:,1],dtype=np.float64)

        self.tolerance = TOLERANCE #tolerance for solution
        self.stepTime = STEP_TIME #seconds
        self.time = START_TIME
        self.fuel_level = FUEL_LEVEL_INITIAL
        self.fuel_expended = 0
        self.throttle = 0
        self.mass = FUEL_LEVEL_INITIAL + 2000  # spacecraft mass is 2000 kg plus 10k kg fuel
        # circular orbit about earth: initial conditions
        r_earth = getPlanetVecElems(BODY_DICT['earth'].id, self.time)[0]
        v_earth = getPlanetVecElems(BODY_DICT['earth'].id, self.time)[1]
        # r_earth, v_earth = planetPositionVelocity(BODY_DICT['earth'].id, self.time)
        self.position = r_earth + INIT_POS  # position relative to Sun
        self.velocity = v_earth + INIT_VEL  # velocity relative to Sun
        self.orientation = np.array([0, 1, 0])  # orientation relative to central body frame
        self.done = False
        self.reward = 0  # initialize reward
        self.state = None
        
    def reset(self):
        # reset the environment state to an initial state, in orbit about Earth
        self.time = START_TIME
        self.fuel_level = FUEL_LEVEL_INITIAL
        self.fuel_expended = 0
        self.throttle = 0
        self.mass = FUEL_LEVEL_INITIAL + 2000  # spacecraft mass is 2000 kg plus 10k kg fuel
        # circular orbit about earth: initial conditions
        r_earth = getPlanetVecElems(BODY_DICT['earth'].id, self.time)[0]
        v_earth = getPlanetVecElems(BODY_DICT['earth'].id, self.time)[1]
        self.position = r_earth + INIT_POS  # position relative to Sun
        self.velocity = v_earth + INIT_VEL  # velocity relative to Sun
        self.orientation = np.array([0, 1, 0])  # orientation relative to central body frame
        self.done = False
        self.reward = 0  # initialize reward
        self.observation = self.state
        
        self.state = np.hstack([getPlanetVecElems(BODY_DICT[planet].id, self.time) for planet in BODY_DICT.keys()]).T
        # self.state = (getPlanetVecElems(BODY_DICT['earth'][.id], self.time)[0][0],
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0][1],
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0][2],
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               getPlanetVecElems(BODY_DICT['earth'][1], self.time)[0]
        #               )

    def spacecraftEOM(self, t, y0, Fnet):
        return [y0[3],
                y0[4],
                y0[5],
                Fnet[0]/self.mass,
                Fnet[1]/self.mass,
                Fnet[2]/self.mass]


    def body_force_on_spacecraft(self, body_mu, body_position, sc_position):
        #body_position is 3x8
        #sc_position is 3x1
        # Fg = mu*r/norm(r)^3 *(rhat)
        return body_mu*(body_position-sc_position)/np.linalg.norm(body_position - sc_position) ** 3
        # return sum( mu(i)*(current_sc_position - body_position(i))/norm(current_sc_position - body_position(i))^3)
        # return np.sum(np.divide(np.multiply(BODY_DICT[x]['mu'][0] for x in BODY_DICT.keys() * (body_position[y] for y in body_position)),
        #                  np.linalg.norm(np.subtract(sc_position, (body_position[z] for z in body_position)) ** 3)))

    def spend_fuel(self, amt):
        self.fuel_expended += amt
        self.mass -= amt
        self.fuel_level -= amt

        if self.fuel_level <= 0:
            self.done = True
        else:
            self.done = False


    def step(self, action):
        # execute one time step within the environment
        # update observations
        self.time += dt.timedelta(minutes=self.stepTime/60)

        planet_positions = getPlanetPositionArray(self.time)
        sc_position = self.position

        # compute sum of forces acting on spacecraft
        Fnet = np.sum([self.body_force_on_spacecraft(planet_position,
                                                     sc_position)
                       for planet_position in planet_positions])

        if self.throttle == 1 & self.fuel_level > 0:
            self.spend_fuel(M_DOT * self.stepTime) #multiply mass flow rate (kg/s) by step-time of fuel burned
            # get the direction of SC velocity as a unit vector
            # print('before syntax error')
            # print(self.orientation) 
            # print(np.linalg.norm(self.orientation) )
            scDirection = self.orientation / np.linalg.norm(self.orientation) #[0 1 0] / 1
            Fnet += MAIN_ENGINE_THRUST * scDirection  # add thrust to Fnet
        else:
            self.throttle = 0
      
        # perform a step-wise integration to compute spacecraft new position and velocity
        newSCStateVector = integrate.solveivp(self.spacecraftEOM([0, 1],
                                                                 [self.position, self.velocity],
                                                                 Fnet))
        self.position = newSCStateVector[0]
        self.velocity = newSCStateVector[1]
        
        # MW - should this really be here and not earlier?
        self._take_action(action)
        self.current_step += 1

        # reward changes depending on optimization goal, for now optimize for fuel consumption
        targetid = '301' #the target object to orbit
        targetRange = 50 #the target circular orbital altitude
    def reward(self, reward,fuel_expended,position,velocity,targetid,time,targetRange,tolerance):
        rew = self.reward
        fe = self.fuel_expended
        r = self.position
        t = self.time
        v = self.velocity
        tol = self.tolerance
        a = rew - fe - abs(np.linalg.norm(np.subtract(r,getPlanetVecElems(targetid,t)[0]) - targetRange+BODY_DICT['moon'][2]))
        # self.reward = self.reward - self.fuel_expended - abs(np.linalg.norm(np.subtract(self.position, getPlanetVecElems(targetid,self.time)[0]) - targetRange+BODY_DICT['moon'][2]))  
        
        if abs(np.linalg.norm(r-getPlanetVecElems(targetid,d)[0]) - (targetRange+BODY_DICT['moon'][2]) ) <= tol and:
             
            (targetRange+BODY_DICT['moon'][2])
        abs(np.linalg.norm(v) - np.linalg.norm(getPlanetVecElems(targetid,d)[1]) <= tol:
            self.done = True
            
        # if abs(np.linalg.norm(self.position - getPlanetVecElems(targetid,self.time)[0]) - targetRange+BODY_DICT['moon'][2]) <= self.tolerance \
        #         and abs(np.linalg.norm(self.velocity) - np.linalg.norm(getPlanetVecElems(targetid,self.time)[1])) <= self.tolerance:
        #     self.done = True

    def _take_action(self, action):
        action_type = action[0]
        if action_type == 1:
            self.throttle = 1
        elif action_type == 2:
            alpha = 1
            self.throttle = 0
            self.orientation @= RotationMatrix(alpha, 0, 0)
        elif action_type == 3:
            alpha = -1
            self.throttle = 0
            self.orientation @= RotationMatrix(alpha, 0, 0)
        elif action_type == 4:
            beta = 1
            self.throttle = 0
            self.orientation @= RotationMatrix(0, beta, 0)
        elif action_type == 5:
            beta = -1
            self.throttle = 0
            self.orientation @= RotationMatrix(0, beta, 0)
        elif action_type == 6:
            gamma = 1
            self.throttle = 0
            self.orientation @= RotationMatrix(0, 0, gamma)
        elif action_type == 7:
            gamma = -1
            self.throttle = 0
            self.orientation @= RotationMatrix(0, 0, gamma)
        elif action_type == 8:
            # do nothing, coast
            self.throttle = 0



def getPlanetVecElems(planet,prevTime,step='1m'): #defaults to step time of 1 minute
   # planet = '499'
   # planet argument must be a string
   endTime = prevTime + dt.timedelta(seconds=1) 
   #convert times to string format
   TS_start = prevTime.strftime('%Y-%m-%d %H:%M:%S')
   TS_end = endTime.strftime('%Y-%m-%d %H:%M:%S')
   # stepTime = dt.datetime.strptime(time)
   stepTime = {'start':TS_start,'stop':TS_end,'step':step}
   
   obj = Horizons(id_type='id',id=planet,location='@sun',epochs=stepTime)
   vec = Table(obj.vectors())
   vectorDict = {}
   vectorDict["r"] = list(vec.as_array(keep_byteorder=False,names='x,y,z'))
   vectorDict["v"] = list(vec.as_array(keep_byteorder=False,names='vx, vy, vz'))
   
   # vectorDict["zval"] = list(vec.as_array(keep_byteorder=False,names='z, vz'))
   # print(vec[0])
   # return vec.as_array(keep_byteorder=False,names='x,y,z,vx,vy,vz')
   k = vectorDict['r']
   j = vectorDict['v']

   r = np.array([k[0][0],k[0][1],k[0][2]])
   v = np.array([j[0][0],j[0][1],j[0][2]])
   # return np.multiply(q,1.496e8)
   # return vectorDict
   return np.multiply(r,1.496e8), np.multiply(v,1.731e3) #convert from AU to km, au/day to km/s


def getPlanetPositionArray(time):
    planet_ids = [BODY_DICT[x][1] for x in BODY_DICT.keys()]
    return np.array([planetPositionVelocity(planet_ids, time)[0] for planet_ids in planet_ids])

def planetPositionVelocity(planet, julianDate):
    obj = Horizons(id_type='id',id=planet,location='@sun',epochs=julianDate)
    vec = Table(obj.vectors())
    r = vec.as_array(keep_byteorder=False,names='x,y,z').data
    v = vec.as_array(keep_byteorder=False,names='vx, vy, vz').data
    r.dtype=float
    v.dtype=float
    return r * 1.496e8, v * 1.731e3 #convert from AU to km, au/day to km/s

if __name__ == '__main__':
    print('hello world')
    SCE = SpaceCraftEnv()
    r_earth = getPlanetVecElems(BODY_DICT['earth'].id, START_TIME)[0]
    print(SCE.body_force_on_spacecraft(BODY_DICT['earth'].mu, r_earth, INIT_POS))
    
# print(getPlanetVecElems('499','00:00')['r'])
# print(getPlanetVecElems('499','00:00')['v'])

# #indexing
# xpos = getPlanetVecElems('499','00:00')['xval'][0][0]
# xvel = getPlanetVecElems('499','00:00')['xval'][0][1]
# print(xpos)
# print(xvel)
