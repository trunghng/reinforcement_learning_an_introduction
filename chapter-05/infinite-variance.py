import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Environment:

    def __init__(self):
        self.ACTIONS = {'left': 0, 'right': 1}
        self.REWARD0 = 0
        self.REWARD1 = 1
        self.STATES = {'non-terminal': 0, 'terminal': 1}


    def take_action(self, state, action):
        if action == self.ACTIONS['left']:
            if np.random.binomial(1, 0.9) == 1:
                next_state = self.STATES['non-terminal']
                reward = self.REWARD0
            else:
                next_state = self.STATES['terminal']
                reward = self.REWARD1
        else:
            next_state = self.STATES['terminal']
            reward = self.REWARD0
        return next_state, reward


    def is_terminal(self, state):
        return state == self.STATES['terminal']


def behavior_policy():
    return np.random.binomial(1, 0.5) == 1


def target_policy(env):
    return env.ACTIONS['left']


def play(env):
    state = env.STATES['non-terminal']
    trajectory = []

    while not env.is_terminal(state):
        action = behavior_policy()
        next_state, reward = env.take_action(state, action)
        trajectory.append([state, action, reward])
        state = next_state
    return trajectory


if __name__ == '__main__':
    env = Environment()
    runs = 10
    episodes = 100000

    for run in tqdm(range(runs)):
        returns = []
        for ep in range(episodes):
            trajectory = play(env)

            if trajectory[-1][1] != target_policy(env):
                rho = 0
            else:
                rho = pow(1 / 0.5, len(trajectory))

            # since only value of the last reward matters
            returns.append(rho * trajectory[-1][2])
        returns = np.cumsum(returns)
        V = returns / np.arange(1, episodes + 1)
        plt.plot(V)
    plt.axhline(y=1, color='black', linestyle='dashed')
    plt.axhline(y=2, color='black', linestyle='dashed')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('MC estimate of value function with OIS')
    plt.xscale('log')
    plt.savefig('./infinite_variance.png')
    plt.close()