import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
import math

NO_STATES = 19
# states with value 0, 20 are 2 dummy states, indicate 2 terminal states.
STATES = {c: i for c, i in zip(string.ascii_uppercase[:NO_STATES + 1],range(1,  NO_STATES + 1))}
START_STATE = STATES['J']
TRUE_VALUES = np.arange(-20, 22, 2) / 20.0
TRUE_VALUES[0] = TRUE_VALUES[-1] = 0


def is_termial(state):
    return state == 0 or state == NO_STATES + 1


def take_action(state, action):
    next_state = state + action
    if next_state == 0:
        reward = -1
    elif next_state == NO_STATES + 1:
        reward = 1
    else:
        reward = 0
    return next_state, reward


def n_step_temporal_difference(V, n, alpha, gamma):
    state = START_STATE
    T = float('inf')
    t = 0

    states = []
    states.append(state)
    rewards = []
    rewards.append(0) # dummy reward to save the next reward as R_{t+1}

    while True:
        if t < T:
            # not agent's action
            action = np.random.choice([-1, 1])
            next_state, reward = take_action(state, action)
            states.append(next_state)
            rewards.append(reward)
            if is_termial(next_state):
                T = t + 1
        tau = t - n + 1 # updated state's time
        if tau >= 0:
            G = 0 # return
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += math.pow(gamma, i - tau - 1) * rewards[i]
            if tau + n < T:
                G += math.pow(gamma, n) * V[states[tau + n]]
            if not is_termial(states[tau]):
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

    errors = np.zeros((len(ns), len(alphas)))
    for n_i, n in enumerate(ns):
        for alpha_i, alpha in enumerate(alphas):
            for _ in tqdm(range(runs)):
                V = np.zeros(NO_STATES + 2)
                for _ in range(episodes):
                    n_step_temporal_difference(V, n, alpha, gamma)
                    rmse = np.sqrt(np.sum(np.power(V - TRUE_VALUES, 2) / NO_STATES))
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





