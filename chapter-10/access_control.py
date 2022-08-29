import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from tile_coding import IHT, tiles

class AccessControl:

    def __init__(self, n_servers):
        self.n_servers = n_servers
        self.rewards = [1, 2, 4, 8]
        self.action = [-1, 1]


    def take_action(self, state, action):






class ValueFunction:

    def __init__(self, n_tilings):
        self.n_tilings = n_tilings
        self.w = np.zeros(2048)
        self.iht = IHT(2048)


    def get_active_tiles(self, state, action):
        '''
        Get active tiles

        Params
        ------
        '''
        active_tiles = tiles(self.iht, self.n_tilings, [state], [action])
        return active_tiles


    def get_value(self, state, action):
        '''
        Get value
        '''
        active_tiles = self.get_active_tiles(state, action)
        return np.sum(self.w[active_tiles])


    def learn(self, state, action, target, alpha):
        '''
        Update weight vector

        Params
        ------
        state:
        '''
        active_tiles = self.get_active_tiles(state, action)
        estimate = np.sum(self.w[active_tiles])
        error = target - estimate
        for tile in active_tiles:
            self.w[tile] += alpha * error


def epsilon_greedy(epsilon, value_function, env, state):
    pass


def differential_semi_gradient_sarsa(value_function, env, alpha, beta, gamma, epsilon):
    '''
    Differenntial Semi-gradient Sarsa algorithm

    Params
    ------
    value_function: ValueFunction
    env: AccessControl
    alpha: float
        step size param
    gamma: float
        discount factor
    epsilon: float
        epsilon greedy param
    '''
    pass


if __name__ == '__main__':
    alpha = 0.01
    beta = 0.01
    gamma = 1
    epsilon = 0.1




