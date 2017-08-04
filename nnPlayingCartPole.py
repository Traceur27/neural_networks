import gym
import random
import numpy as np
#import keras

from statistics import mean, median
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


learningRate = 1e-3

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirement = 50
initial_games = 10000


def play_random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

play_random_games();