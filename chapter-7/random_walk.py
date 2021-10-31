import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


class RandomWalk:

    def __init__(self, n_states, start_state):
        self._n_states = n_states
        self._states = np.arange(1, n_states + 1)
        self._start_state = start_state
        self._end_states = [0, n_states + 1]
        self.ACTION_LEFT = -1
        self.ACTION_RIGHT = 1
        self._actions = [self.ACTION_LEFT, self.ACTION_RIGHT]
        self._action_prob = 0.5


    @property
    def n_states(self):
        return self._n_states


    @property
    def states(self):
        return self._states


    @property
    def start_state(self):
        return self._start_state


    @property
    def end_states(self):
        return self._end_states


    @property
    def actions(self):
        return self._actions


    @property
    def action_prob(self):
        return self._action_prob
    

    def is_terminal(self, state):
        return state in self.end_states


    def take_action(self, state, action):
        next_state = state + action
        if next_state == 0:
            reward = -1
        elif next_state == self.n_states + 1:
            reward = 1
        else:
            reward = 0
        return next_state, reward



def get_true_value(random_walk, gamma):
    """
    Compute true value by Bellman equations
    """
    P = np.zeros((random_walk.n_states, random_walk.n_states))
    r = np.zeros((random_walk.n_states + 2, ))
    true_value = np.zeros((random_walk.n_states + 2, ))
    
    for state in random_walk.states:
        next_states = []
        rewards = []

        for action in random_walk.actions:
            next_state = state + action
            next_states.append(next_state)

            if next_state == 0:
                reward = -1
            elif next_state == random_walk.n_states + 1:
                reward = 1
            else:
                reward = 0
            rewards.append(reward)

        for state_, reward_ in zip(next_states, rewards):
            if not random_walk.is_terminal(state_):
                P[state - 1, state_ - 1] = random_walk.action_prob * 1
                r[state_] = reward_
        
    u = np.zeros((random_walk.n_states, ))
    u[0] = random_walk.action_prob * 1 * (-1 + gamma * -1)
    u[-1] = random_walk.action_prob * 1 * (1 + gamma * 1)

    r = r[1:-1]
    true_value[1:-1] = np.linalg.inv(np.identity(random_walk.n_states) - gamma * P).dot(0.5 * (P.dot(r) + u))
    true_value[0] = -1
    true_value[-1] = 1

    return true_value


def get_action(random_walk):
    return np.random.choice(random_walk.actions)


def n_step_temporal_difference(V, n, alpha, gamma, random_walk):
    state = random_walk.start_state
    T = float('inf')
    t = 0

    states = []
    states.append(state)
    rewards = []
    rewards.append(0) # dummy reward to save the next reward as R_{t+1}

    while True:
        if t < T:
            action = get_action(random_walk)
            next_state, reward = random_walk.take_action(state, action)
            states.append(next_state)
            rewards.append(reward)
            if random_walk.is_terminal(next_state):
                T = t + 1
        tau = t - n + 1 # updated state's time
        if tau >= 0:
            G = 0 # return
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += math.pow(gamma, i - tau - 1) * rewards[i]
            if tau + n < T:
                G += math.pow(gamma, n) * V[states[tau + n]]
            if not random_walk.is_terminal(states[tau]):
                V[states[tau]] += alpha * (G - V[states[tau]])
        t += 1
        if tau == T - 1:
            break
        state = next_state
        

def rmse():
    episodes = 10
    runs = 100
    gamma = 1
    ns = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    random_walk = RandomWalk(19, 10)
    true_value = get_true_value(random_walk, gamma)

    errors = np.zeros((len(ns), len(alphas)))
    for n_i, n in enumerate(ns):
        for alpha_i, alpha in enumerate(alphas):
            for _ in tqdm(range(runs)):
                V = np.zeros(random_walk.n_states + 2)
                for _ in range(episodes):
                    n_step_temporal_difference(V, n, alpha, gamma, random_walk)
                    rmse = np.sqrt(np.sum(np.power(V - true_value, 2) / random_walk.n_states))
                    errors[n_i, alpha_i] += rmse

    errors /= episodes * runs

    for i in range(0, len(ns)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (ns[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()
    plt.savefig('./random_walk.png')
    plt.close()


if __name__ == '__main__':
    rmse()





