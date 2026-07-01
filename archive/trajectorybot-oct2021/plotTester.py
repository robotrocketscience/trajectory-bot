#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:28:59 2021

@author: yoshi
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from collections import namedtuple
# %matplotlib notebook
# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    
    
# MassiveBody = namedtuple(typename='massive_body',
#                          field_names=['mu', 'id','radius'])

# BD = {'sun': MassiveBody(mu=132700000000, id='10',radius=696500),
#              'mercury': MassiveBody(mu=2.2032e4, id='1',radius=2441),
#              'venus':  MassiveBody(mu=3.2485e5, id='299',radius=6051.89),
#              'earth': MassiveBody(mu=398600, id='399',radius=6378),
#              'moon': MassiveBody(mu=4902.8 ,id='301', radius=1738),
#              'mars': MassiveBody(mu=42820, id='499',radius=3396),
#              'jupiter': MassiveBody(mu=1.26686e8, id='599',radius=69911),
#              'saturn': MassiveBody(mu=3.79311e7, id='699',radius=58232),
#              'uranus': MassiveBody(mu=5.7939e6, id='799',radius=25362),
#              'neptune': MassiveBody(mu=6.8365e6, id='899',radius=24624)
#              }
def plot_trajectory(trajectory):
    m = [92347765.81062363, -120857389.89913307, 12551.392607609736]  #moon initial location relative to sun
    e = [9.24385475e+07, -1.20492347e+08,  5.26608509e+03] #earth initial location  relative to sun
    rME = np.subtract(m,e)
    
    
    
    re = 6378 #earth radius
    # Generate and plot a earth size sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = re*np.outer(np.cos(u), np.sin(v)) # np.outer() -> outer vector product
    y = re*np.outer(np.sin(u), np.sin(v))
    z = re*np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, color='g')
    
    rm = 1738 #moon radius
    # Generate and plot a moon size sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = rME[0]+rm*np.outer(np.cos(u), np.sin(v)) # np.outer() -> outer vector product
    y = rME[1]+rm*np.outer(np.sin(u), np.sin(v))
    z = rME[2]+rm*np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='r')
    
    zline = []
    yline = []
    xline = []
    
    # plot the last 10 spacecraft trajectories
    # for i in range(trajectory[-1][3]-1):
        
    for x in range(len(trajectory)-1): 
        zline.append(trajectory[x][2])
        yline.append(trajectory[x][1])
        xline.append(trajectory[x][0])
    
    ax.plot3D(xline,yline,zline)
    
    
    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required
    ax.set_title("Earth-Moon System")
    ax.set_xlabel('x coord, [km]')
    ax.set_ylabel('y coord, [km]')
    ax.set_zlabel('z coord, [km]')
    plt.show()



# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111,projection='3d')

# re = 6378 #earth radius
# u,v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
# x = re*np.cos(u)*np.sin(v)
# y = re*np.sin(u)*np.sin(v)
# z = re*np.cos(v)
# ax.plot_wireframe(x,y,z,color="g")

# rm = 1738 #moon radius
# u,v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
# x = rME[0]+rm*np.cos(u)*np.sin(v)
# y = rME[1]+rm*np.sin(u)*np.sin(v)
# z = rME[2]+rm*np.cos(v)
# ax.plot_wireframe(x,y,z,color="r")


# x = np.append([0], [obs[i] for i in range(0,27,3)])
# y = np.append([0], [obs[i] for i in range(1,27,3)])
# z = np.append([0], [obs[i] for i in range(2,27,3)])
# size = np.array([BD[planet].radius for planet in BD.keys()]) 
# size = size/(1e-1*max(size)/2)

# ax.scatter3D(x,y,z)



# plt.show()



