import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import ShortCorridor


class REINFORCE:
    '''
    REINFORCE algorithm
    '''

    def __init__(self, env: ShortCorridor, 
                alpha: float, 
                gamma: float,
                epsilon: float,
                theta: np.ndarray):
        '''
        Params
        ------
        env: ShortCorridor
        alpha: step size
        gamma: discount factor
        theta: 
        '''
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = theta
        # feature mapping
        self.x = lambda s, a: np.array([int(a == self.env.action_space[1]), 
                                        int(a == self.env.action_space[0])])


    def preference(self, state: int, action: int) -> float:
        '''
        Linear action preferences

        Params
        ------
        state: state of the agent
        action: action taken at state @state
        '''
        feature_vector = self.x(state, action)
        return self.theta.dot(feature_vector)


    def pi(self, state: int, action: int) -> float:
        '''
        Softmax in action preferences

        Params
        ------
        state: state of the agent
        action: action taken at state @state
        '''
        def _softmax(zi, z):
            zmax = np.max(z)
            return np.exp(zi - zmax) / np.sum(np.exp(z - zmax))
        
        action_pref = self.preference(state, action)
        action_prefs = []
        for action_ in self.env.action_space:
            action_prefs.append(self.preference(state, action_))

        return _softmax(action_pref, np.array(action_prefs))


    def select_action(self, state: int, pi: List[List[float]]) -> int:
        amin = np.argmin(pi[state])
        pi[state][amin] = self.epsilon
        pi[state][1 - amin] = 1 - self.epsilon

        action = self.env.action_space[1] \
            if np.random.binomial(1, pi[state][self.env.action_space[1]]) \
            else self.env.action_space[0]

        # if not np.random.binomial(1, self.epsilon):
        #     action = np.random.choice(self.env.action_space)
        # else:
        #     action = self.env.action_space[np.argmax(pi[state])]
        return action


    def eligibility_vector(self, state: int, action: int) -> np.ndarray:
        mean_feature = 0
        for action_ in self.env.action_space:
            mean_feature += self.pi(state, action_) * self.x(state, action_)

        return self.x(state, action) - mean_feature


    def update(self, state: int,
                action: int, return_: float, step: int) -> None:
        eligibility_vector = self.eligibility_vector(state, action)
        self.theta += self.alpha * (self.gamma**step) \
                * return_ * eligibility_vector


    def run(self) -> float:
        '''
        Perform an episode
        '''
        state = self.env.reset()
        trajectory = []
        pi = [[self.pi(state_, action_) for action_ in self.env.action_space] \
                    for state_ in self.env.state_space]
        total_reward = 0

        # Generate episode + compute cumulative rewards
        while True:
            action = self.select_action(state, pi)
            next_state, reward, terminated = self.env.step(action)
            for t in range(len(trajectory)):
                trajectory[t][3] += self.gamma**(len(trajectory) - t) * reward
            trajectory.append([state, action, reward, reward])
            if terminated:
                break
            state = next_state

        for t in range(len(trajectory)):
            state, action, reward, return_ = trajectory[t]
            self.update(state, action, return_, t)
            total_reward += reward

        return total_reward


if __name__ == '__main__':
    n_states = 3
    start_state = 0 
    terminal_state = 3
    switched_states = [1]
    env = ShortCorridor(n_states, start_state, terminal_state, switched_states)
    alphas = [2e-3, 2e-4, 2e-5]
    gamma = 1
    epsilon = 0.05
    n_runs = 100
    n_eps = 1000
    rewards = np.zeros((len(alphas), n_runs, n_eps))
    colors = ['blue', 'red', 'green']
    labels = ['2e-3', '2e-4', '2e-5']

    for alpha_idx, alpha in enumerate(alphas):

        for run in trange(n_runs):
            
            theta = np.array([-1.47, 1.47])
            for ep in range(n_eps):
                reinforce = REINFORCE(env, alpha, gamma, epsilon, theta)
                total_reward = reinforce.run()
                rewards[alpha_idx, run, ep] = total_reward

    for i in range(len(alphas)):
        plt.plot(np.arange(n_eps), rewards[i].mean(axis=0), \
            color=colors[i], label=r'$\alpha=$' + labels[i])

    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('./short-corridor-reinforce.png')
    plt.close()
