import numpy as np
import matplotlib.pyplot as plt

GOAL = 100
states = np.arange(0, GOAL)
rewards = {'terminal': 1, 'non-terminal': 0}
HEAD_PROB = 0.4


def is_terminal(state):
    return state == 0 or state == GOAL


def value_iteration(theta):
    V = np.zeros(states.shape)

    while True:
        delta = 0
        for i in range(V.shape[0]):
            old_value = V[i]

            actions = np.arange(0, min(i, 100 - i) + 1)


if __name__ == '__main__':
    theta = 1e-5
    value_iteration(theta)
