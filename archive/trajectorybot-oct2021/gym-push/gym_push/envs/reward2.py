#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 19:59:17 2021

@author: robotrocketscience
"""

    reward = 0 #initialize reward at 0
    
    #if it crashes punish severely and break out early to avoid extra computing
    if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['moon'].id,self.time)[0])) <= BODY_DICT['moon'].radius:
        reward -= 1000
        self.impact = True
        print(self.position)
    
    if abs(np.linalg.norm(self.position - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0])) <= BODY_DICT['earth'].radius:
        reward -= 1000
        self.impact = True
        print(self.position)
    
    # breakpoint()    
    #if reward gets way too negative just end the trial
    if self.reward < -1e3:
        self.impact = True
    #punish use of fuel in general
    fuel_expended = FUEL_LEVEL_INITIAL - self.fuel_level
    reward += 0.2*fuel_expended**1.2 + 0.3*fuel_expended
    #punish changing orientation in general
    reward -= 10*math.log10(abs(np.linalg.norm(self.orientation-INIT_ORIENT))+.01)
    # reward approaching the target 
    reward += math.exp(1/(0.001+abs(np.linalg.norm(np.subtract(self.position,planetPositionVelocity(BODY_DICT[TARGET_PLANET].id,self.time)[0]) - TARGET_ORBIT+BODY_DICT[TARGET_PLANET].radius))/1e4))
    # breakpoint()
    pi = math.pi
    mu_e =  398600
    mu_m = 4902.8
    rsoi_e = 924000
    rsoi_m = 66100
    
    r_e_sc = self.position - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0] #r from earth to SC
    v_e_sc = self.velocity - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[1] #v of sc relative to earth
    r_e_m = planetPositionVelocity(BODY_DICT['moon'].id,self.time)[0] - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[0] #r from earth to moon
    v_e_m = planetPositionVelocity(BODY_DICT['moon'].id,self.time)[1] - planetPositionVelocity(BODY_DICT['earth'].id,self.time)[1] #v from earth to moon
    r_m_sc = r_e_sc - r_e_m #from moon to spacecraft

    # get initial values for r_e_sc, r_e_m, r_m_sc
    r_e_sc0 = INIT_POS #relative to earth
    r_e_m0 = planetPositionVelocity(BODY_DICT['moon'].id,START_TIME)[0] - planetPositionVelocity(BODY_DICT['earth'].id,START_TIME)[0]
    r_m_sc0 = r_e_sc0 - r_e_m0
    
    # interceptor must lead target by lead angle alpha, when interceptor starts hohmann transfer
    # target angular velocity w
    w = math.sqrt(mu_e/np.linalg.norm(r_e_m)**3)
    a_xfer = np.linalg.norm(r_e_m0)+TARGET_ORBIT+BODY_DICT['moon'].radius
    TOF = pi*math.sqrt(np.linalg.norm(r_e_sc0)**3 /mu_e)
    # TOF = pi*math.sqrt(a_xfer**3 /mu_e)
    alpha_lead = w*TOF #radians
    phi_f = pi-alpha_lead #phase angle between interceptor and target as transfer begins (rads)
    
    #must wait to be in correct position
    phi_i = np.arccos(np.dot(r_e_sc,r_e_m)/(np.linalg.norm(r_e_sc)*np.linalg.norm(r_e_m)))
    w_sc = math.sqrt(mu_e/np.linalg.norm(r_e_sc)**3)
    numer = phi_f - phi_i
    denom = w-w_sc
    wait_time = (numer/denom)/60 #in minutes
    if wait_time < 0:
        wait_time = ((2*pi+numer)/denom)/60
    if wait_time < 0:
        wait_time = ((numer-2*pi)/denom)/60
        
    # breakpoint()
    #STAGE 1, pre-transfer, self.beginxfer = false,false, optimal action is to wait until phi_f - phi_i = 0
    if ( 180*numer/pi <= 2  ) and ( np.linalg.norm(r_e_sc) <= rsoi_e ):
        self.beginxfer[0] = True
        #compute deltaV for lunar transfer
        r_e_sc_startXfer = np.cos(phi_i)*np.linalg.norm(INIT_POS)*np.linalg.norm(r_e_m)*1/r_e_m
        deltaV = 4.04
        # deltaV = math.sqrt(mu_e/np.linalg.norm(r_e_sc_startXfer))*(math.sqrt(2*a_xfer/(np.linalg.norm(r_e_sc_startXfer)+a_xfer))-1)
        print('transfer orbit in progress')

    elif self.beginxfer == [False,False]:
        print('STAGE 1')
        reward += math.exp(np.dot(self.orientation,(self.velocity/np.linalg.norm(self.velocity))))  #reward maintaining prograde orientation
        if self.action == 0:
            reward += 100
        elif self.action > 0:
            reward -= 100

    # breakpoint()
    #STAGE 2, transfer initiated, self.beginxfer = true,false, establish transfer ellipse, optimal action is to thrust prograde
    if self.beginxfer == [True, False]: #set up the ellipse path
        print('STAGE 2')
        print('establishing transfer ellipse and thrusting prograde')
        # zero_time = convertTimeToJulian(self.time)
        end_time = TOF/60 #convert to days to be consistent with Julian time
        t = np.linspace(0,end_time,10000)
        solution = odeint(F,np.append(r_e_sc_startXfer,INIT_VEL+deltaV),t,args=(mu_e,))
        self.x = solution[:,0]
        self.y = solution[:,1]
        self.z = solution[:,2]
        self.lockEllipse = True
        self.beginxfer = [False,True]
        # breakpoint()
        # plt.plot(solution[:,0],solution[:,1], '.')
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(111,projection='3d')
        # ax.plot3D(solution[:,0],solution[:,1],solution[:,2])
        
        if self.action == 0:
            reward -= 5
        elif self.action > 1:
            reward -= 5
        elif self.action == 1: 
            reward += 5
                
    # breakpoint()    
    if self.lockEllipse == True:
        print('STAGE 3')
        print('tracking ellipse')
        delrx = abs(r_e_sc[0] - self.x[self.tracker])
        delry = abs(r_e_sc[1] - self.y[self.tracker])
        delrz = abs(r_e_sc[2] - self.z[self.tracker])
        # breakpoint()
        if self.action == 0:
            reward += 5
        elif self.action > 0:
            reward -= 5
        reward += math.exp(3/(1+delrx+delry+delrz))
        self.tracker += 1
        # breakpoint()
    
    if np.linalg.norm(r_m_sc) <= rsoi_m: #arrived in moon SOI
        #turn off ellipse shit
        self.lockEllipse = False
        print('STAGE 4')
        print('circularizing around target')
        #get rewarded for circularizing orbit
        r_target = TARGET_ORBIT+1738
        delr = np.linalg.norm(r_m_sc) - r_target
        delv = abs(math.sqrt(mu_m/r_target) - np.linalg.norm(v_e_sc-v_e_m))
        if self.action <= 1:
            reward += 5
        elif self.action > 1:
            reward -= 5
        reward += math.exp(3/(1+delr))
        reward += math.exp(3/(1+delv))
        # breakpoint()