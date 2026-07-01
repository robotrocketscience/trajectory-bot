#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:23:05 2021

@author: yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def plot_trajectory_curve(time,distance,figure_file,n):
    
    plt.plot(time,distance,label = 'Spacecraft Trajectory')

    plt.xlabel('time, minutes')
    plt.ylabel('distance, [km]')
    plt.title('Distance Between Spacecraft and Target, trial ' + str(n))
    plt.savefig(figure_file)
#def telnetFetcher()