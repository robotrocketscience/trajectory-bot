#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:23:25 2021

@author: robotrocketscience
"""
import gym
from ppo_torch import Agent
from utils import plot_learning_curve
import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# import basic_env1
# import basic_env1
# import TrajectoryBot_V2

if __name__ == '__main__':
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'basic-v0' in env:
              print("Remove {} from registry".format(env))
              del gym.envs.registration.registry.env_specs[env]
    from spacecraft.basic_env1 import SpaceCraftEnv
                
    env = gym.make('gym_push:basic-v0')
    # env = SpaceCraftEnv()
    
    N = 5
    batch_size = 64 #think about making this bigger, like 64
    n_epochs = 4
    alpha = 2
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)

    n_games = 1
    
    figure_file = 'plots/MoonTrajectory.png'

    best_score = env.reward_range[0]
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    # trajectory = []
    scpos = []
    
    #load previous training
    # agent.load_models()
    # m = [0,0,0]
    # sc_loc = [np.array([0,0,0])]
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        cnt = 1
        time = 1
        # m.append([observation[9],observation[10],observation[11]])
        while not done:
            time += 1
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
                                                    
            if n_steps % N == 0:
                # breakpoint()
                agent.learn()
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
    
    # columnNames = [['sc x','sc y', 'sc z','e x','e y','e z','m x','m y','m z','alpha','beta','gamma','fuel','throttle','trial']]  
    # file  = open('data.csv','w+',newline='')
    # with file:
    #     write = csv.writer(file)
    #     write.writerows(columnNames)
    # file = open('data.csv','a+',newline='')
    # with file:
    #     write = csv.writer(file)
    #     write.writerows(trajectory)
        
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # scx = [];
    # scy = [];
    # scz = [];
    # for i in range(len(scpos)):
    #     scx.append(scpos[i][0])
    #     scy.append(scpos[i][1])
    #     scz.append(scpos[i][2])
    # ax.plot3D(scx,scy,scz)
    
