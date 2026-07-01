#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:36:12 2021

@author: robotrocketscience
"""
from TBot_Inputs import *
               

def planetPositionVelocity(planet,start,stop,numsteps):
    obj = Horizons(id_type='id',id=planet,location='@sun',
                   epochs={'start':start.iso,
                           'stop':stop.iso,
                           'step':str(numsteps)})
    vec = Table(obj.vectors())
    r = np.array(vec['x','y','z'])
    v = np.array(vec['vx','vy','vz'])
    r.dtype=float
    v.dtype=float
    # print('planet pos vel')
    return ((r*1.496e8), (v*1.731e3)) #convert from au to km, au/day to km/s


def getBodyAccel(self,planet):
    #outputs the gravitational acceleration imparted by a planet 
    mu = BODY_DICT[planet].mu
    r_body = self.ed[planet].r[0][self.stepNum*3:self.stepNum*3+3]
    # print('get body force')
    return -mu*(self.position-r_body)/(np.linalg.norm(self.position-r_body)**3)

def spacecraftEOM(self,accel,time):
    # time = time.total_seconds()
    time = time.to_value('s')
    # rescale Fnet by 1000
    v = self.velocity + accel*time
    # print(Fnet)
    r = self.position + self.velocity*time + 0.5*accel*time**2
    # print('r: ',r)
    # print('v: ',v)
    # print('sc eom')
    # print('Fnet '+str(np.linalg.norm(Fnet)))
    return r,v

def generateEphDict(st,et,numsteps):
    ephem = namedtuple(typename='planet',field_names=['r','v'])

    for planet in range(10): 
        body = list(BODY_DICT.keys())[planet]
        if planet == 0:
            EPH_DICT = {body: ephem(
                r=[planetPositionVelocity(BODY_DICT[body].id,st,et,numsteps)[0]],
                v=[planetPositionVelocity(BODY_DICT[body].id,st,et,numsteps)[1]])}
        else:
            EPH_DICT.update({body: ephem(
                r=[planetPositionVelocity(BODY_DICT[body].id,st,et,numsteps)[0]],
                v=[planetPositionVelocity(BODY_DICT[body].id,st,et,numsteps)[1]])})            
    return EPH_DICT

def observeState(self):
    #spacecraft values to keep track of as part of the state vector
    scstate = np.hstack([self.fuelMass,
                         self.throttle,
                         self.orientation,
                         self.reward,
                         self.position,
                         self.velocity
                         ])
    #organize planet position and velocity vector elements in one column
    #if we have a new ephemeris dictionary
    if self.stepNum == 0: 
        #initialize r and v stacks
        r=[]
        v=[]
        iterbody = iter(BODY_DICT.keys())
        next(iterbody) #skip sun
        for planet in iterbody:
            r = np.append(r,np.hstack(self.ed[planet].r))
            v = np.append(v,np.hstack(self.ed[planet].v))

        #update hour-forward observations
        self.pp = np.hstack(r)
        self.vp = np.hstack(v)
        
    return np.hstack([scstate,self.pp,self.vp])
    
     
def defineObsSpace(self):
    #count the number of observations
    numObs = sum([INIT_VEL.shape[0], #velocity 3
                  INIT_POS.shape[0], #position 3
                  INIT_ORIENT.shape[0], #orientation 3
                  1,#FUEL_LEVEL_INITIAL is a float
                  1,#throttle is a float
                  1 #reward is a float
                  ])
    iterbody = iter(BODY_DICT.keys())
    next(iterbody) #skip the sun
    for planet in iterbody:
        numObs += (self.ed[planet].r[0].shape[0])
        numObs += (self.ed[planet].v[0].shape[0])
    
    # initialize observation space having numObs rows with float('inf') values
    obs = np.full((numObs,2),18e9)
    #fuel level min and max
    obs[0][0] = 0
    obs[0][1] = FUEL_LEVEL_INITIAL
    #throttle min and max, percentage
    obs[1][0] = 0
    obs[1][1] = 1
    #orientation is a 1x3 unit vector
    for x in range(2,5):
        obs[x][0] = 0
        obs[x][1] = 1
    #everything else needs to be between -inf and +inf
    for x in range(6,len(obs)):
        obs[x][0] = 18e9
    
    return obs
    
def getReward(self,action):
    reward = 1
    return reward

def checkDone(self):
    done = False
    return done




##################### OLD STUFF, MAY NOT NEED




# def RotationMatrix(a, b, c):
#     return np.array([[np.cos(a) * np.cos(b),
#                       np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c),
#                       np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)],
#                      [np.sin(a) * np.cos(b),
#                       np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
#                       np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)],
#                      [-np.sin(b),
#                       np.cos(b) * np.sin(c),
#                       np.cos(b) * np.cos(c)]])

# def convertTimeToJulian(dtObj):
#     q=dateutil.parser.parse(dt.datetime.strftime(dtObj,'%Y-%m-%d %H:%M:%S'))
#     time = astropy.time.Time(q)
#     return time.jd

# def checkDone(self):
#     #define distance from body to sc for easier book keeping
#     rvec = self.position - planetPositionVelocity(BODY_DICT[self.target].id,self.time)[0]
#     # define circular velocity of SC about target for easier bookkeeping
#     vvec = self.velocity - planetPositionVelocity(BODY_DICT[self.target].id,self.time)[1]
#     targetV = math.sqrt(BODY_DICT[TARGET_PLANET].mu/(TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius))
#     targetT = 2*math.pi*math.sqrt((TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius)**3 /BODY_DICT[TARGET_PLANET].mu)
#     targetE = -BODY_DICT[TARGET_PLANET].mu/(2*TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius)
#         # compute these values for the spacecraft, relative to the target
#     sc_vel1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[0]
#     sc_T1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[1]
#     sc_E1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[2]
#         # no fuel left is DONE
#     if self.impact == True:
#         done = True
#         print(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0]))
#         print(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT[self.target].id,self.time)[0]))
#         print('crashed')
#     elif self.fuel_level <= 0:
#         done = True   
#         print('out of fuel')
#     #achieve the desired orbit is DONE
#     elif (abs(targetV-sc_vel1) <= self.tolerance) and (abs(targetT-sc_T1) <= self.tolerance) and (abs(targetE-sc_E1) <= self.tolerance):
#         done = True
#         print('orbit achieved')
#     # elif self.reward < -1e15:
#         # done = True
#     else:    
#         done = False
#     #crash into a body is DONE, come back to this condition later
#     # for planet in BODY_DICT.keys():
#     #     if np.linalg.norm(planetPositionVelocity(BODY_DICT[planet].id,self.time)[0]) <= np.linalg.norm(self.position): 
#     #         done = True
#     #         break
#     return done
    

# def observeState(self):
#     #organize planet position vectors in one column, descending order from sun to neptune
#     planet_positions = np.hstack([planetPositionVelocity(BODY_DICT[planet].id, self.time)[0] for planet in BODY_DICT.keys()]).T
#     temp = np.delete(planet_positions,[0,1,2]) #delete sun entries as they are 0
#     temp = np.append(temp,self.position) #append SC position to the state
#     temp = np.append(temp,self.velocity) #append SC velocity to the state
#     temp = np.append(temp,self.orientation) #append orientation
#     temp = np.append(temp,self.fuel_level) #append fuel level
#     temp = np.append(temp,self.throttle) #append throttle (true/false)
#     temp = np.append(temp,self.action)
#     temp = np.append(temp,convertTimeToJulian(self.time))
#     return temp

# # def F(s,t,G): return [s[3],s[4],s[5],
# #     -G*s[0]/(s[0]**2 + s[1]**2 + s[2]**2)**(3/2),
# #     -G*s[1]/(s[0]**2 + s[1]**2 + s[2]**2)**(3/2),
# #     -G*s[2]/(s[0]**2 + s[1]**2 + s[2]**2)**(3/2),
# # ]

# def getReward(self,target,desired_orbit,action):
#     reward = 0 #initialize reward at 0
    
#     #if it crashes punish severely and break out early to avoid extra computing
#     if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['moon'].id,self.time)[0])) <= BODY_DICT['moon'].radius:
#         reward -= 1000
#         self.impact = True
#         print(self.position)
    
#     if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0])) <= BODY_DICT['earth'].radius:
#         reward -= 1000
#         self.impact = True
#         print(self.position)
    
#     # breakpoint()    
#     #if reward gets way too negative just end the trial
#     if self.reward < -1e3:
#         self.impact = True
#     #punish use of fuel in general
#     fuel_expended = FUEL_LEVEL_INITIAL - self.fuel_level
#     reward += 0.2*fuel_expended**1.2 + 0.3*fuel_expended
#     #punish changing orientation in general
#     reward -= 10*math.log10(abs(np.linalg.norm(self.orientation-INIT_ORIENT))+.01)
    

#     return reward


# def getOrbitParams(state,time,target):
#     r_m_sc = state - planetPositionVelocity(BODY_DICT[target].id,time)[0] #position vector of spacecraft relative to moon
#     # v_m_sc = self.velocity - planetPositionVelocity(BODY_DICT[target].id,self.time)[1] #velocity vector of spacecraft relative to moon
        
#     currentV = math.sqrt(BODY_DICT[target].mu/(np.linalg.norm(r_m_sc)+BODY_DICT[target].radius))
#     currentT = 2*math.pi*math.sqrt(((np.linalg.norm(r_m_sc)+BODY_DICT[target].radius)**3 /BODY_DICT[target].mu))
#     currentE = -BODY_DICT[target].mu / (2 * (np.linalg.norm(r_m_sc) + BODY_DICT[target].radius))
#     return currentV,currentT,currentE,r_m_sc