#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:35:26 2021

@author: yoshi
"""

# def getReward(self,target,desired_orbit,action):
#         #a circular orbit is defined by:
#         #speed v = sqrt(mu/r) for all t
#         # period T = 2*pi*sqrt(r^3 /mu)
#         #orbit energy E = -mu/(2*r)
#         #compute the target 
#     targetV = math.sqrt(BODY_DICT[target].mu/(TARGET_ORBIT+BODY_DICT[target].radius))
#     targetT = 2*math.pi*math.sqrt((TARGET_ORBIT+BODY_DICT[target].radius)**3 /BODY_DICT[target].mu)
#     targetE = -BODY_DICT[target].mu/(2*TARGET_ORBIT+BODY_DICT[target].radius)
#     targetVec = planetPositionVelocity(BODY_DICT[TARGET_PLANET].id, self.time)[0] #distance from sun to moon
#         # compute these values for the spacecraft, relative to the target
#     sc_vel1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[0]
#     sc_T1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[1]
#     sc_E1 = getOrbitParams(self.position,self.time,TARGET_PLANET)[2]
#     r_sc_m = targetVec - self.position #vector distance from spacecraft to moon

#     reward = 0
#     # delr = self.position - planetPositionVelocity(BODY_DICT[target].id,self.time)[0]
#     # delv = self.velocity - planetPositionVelocity(BODY_DICT[target].id,self.time)[1]
#     # k = math.sqrt(delr[0]**2 + delr[1]**2 +delr[2]**2 + delv[0]**2 + delv[1]**2 +delv[2]**2)
#     # scaleFactor = 1e-5 
#     # psi = FUEL_LEVEL_INITIAL
#     # if np.linalg.norm(delr) < np.linalg.norm(planetPositionVelocity(BODY_DICT[target].id,self.time)[0]) and np.linalg.norm(delv) < np.linalg.norm(planetPositionVelocity(BODY_DICT[target].id,self.time)[1]):
#     #     reward = .5*psi/(1+self.total_action_count)*math.exp(-scaleFactor*k)
#     # else:
#     #     reward = -10
#     # print('del_reward: ' + str(reward))
        
#     #orbital parameter achievement awards
#     reward += (1-(sc_vel1/targetV)**.2)
#     reward += .1*(1-(sc_T1/targetT)**.2)
#     reward += (1-(sc_E1/targetE)**.2)
#     #arrival condition met bonus
#     if abs(sc_vel1-targetV) <= self.tolerance:
#         reward += 100
#     if abs(sc_T1-targetT) <= self.tolerance:
#        reward += 100
#     if abs(sc_E1 - targetE) <= self.tolerance:
#         reward += 100
        
#     #distance reward
#     reward += .1*(1-(np.linalg.norm(self.position-planetPositionVelocity(BODY_DICT[target].id,self.time)[0])+BODY_DICT[target].radius)**.2)
    
#     #impact penalty
#     if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT[target].id,self.time)[0])) <= BODY_DICT[target].radius:
#         reward -= -1000
#     if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0])) <= BODY_DICT['earth'].radius:
#         reward -= -1000
        
#     #punish changing orientation so damn much
#     if action > 1:
#         reward -= 2
#     # if not np.array_equal(self.orientation,INIT_ORIENT):
#     #     reward -= 5*(abs(self.orientation[0]-INIT_ORIENT[0]) + abs(self.orientation[1]-INIT_ORIENT[1]) + abs(self.orientation[2]-INIT_ORIENT[2]))
    
#     #punish using fuel
#     fuel_expended = FUEL_LEVEL_INITIAL - self.fuel_level
#     reward -= 10*math.log10(fuel_expended+1)
#     if self.fuel_level <= 0:
#         reward -= 100
    
#     # prevTargDistance = np.linalg.norm(self.prevPos-planetPositionVelocity(BODY_DICT[target].id, self.time-dt.timedelta(minutes=1))[0])
#     # currTargDistance = np.linalg.norm(self.position-planetPositionVelocity(BODY_DICT[target].id, self.time)[0])
#     # if currTargDistance >= prevTargDistance:
#     #     reward += 10
#     #     if action == 0:
#     #         reward -= 10
            
#     # reward increasing velocity to moon orbit vel and closing distance to moon
#     # velocity_reward = 10*(1 - (sc_vel1/targetV)**0.2)
#     # distance_reward = 10*(math.log10(np.linalg.norm(r_sc_m)/BODY_DICT[target].radius+desired_orbit))
#     # reward += velocity_reward + distance_reward
#     # print('earth-relative vel ' +str(np.linalg.norm(self.velocity - planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1])))
#     # if np.linalg.norm(self.velocity - planetPositionVelocity(BODY_DICT['earth'].id, self.time)[1]) >= 7.81409+3.02:
#     #     reward -= 10*math.log10(fuel_expended+1)
#     #maybe add earth escape velocity as a reward condition
#     #if no more thrust is needed, (ie reaches correct speed) punish heavily for thrusting
#     # self.velocity > 
    
#     #punish lowering speed wrt earth, punish lowering orbit altitude wrt earth
#     # if np.lingalg.norm(self.velocity)
    
#     return reward 