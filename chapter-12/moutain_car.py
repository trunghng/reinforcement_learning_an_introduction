import sys
import gym
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))

import numpy as np
import matplotlib.pyplot as plt
from tile_coding import tiles, IHT


class ValueFunction:

    def __init__(self):
        pass


    def get_value(self, position, velocity, action):
        pass


    def learn(self):
        pass


def sarsa_lambda(value_function, env, lambda_, alpha, gamma):
    '''
    Sarsa(lambda) algorithm

    Params
    ------
    value_function: ValueFunction
        value function
    env: MountainCar env
    lambda_: float
        trace decay param
    alpha: float
        step size
    gamma: float
        discount factor
    '''
    pass


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.reset()

    runs = 30
    episodes = 50
    alphas = np.arange(1, 8) / 4.0
    lams = [0.99, 0.95, 0.5, 0]

