import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))

import numpy as np

from env import GridWorld


def iterative_policy_evaluation(env: GridWorld, gamma: float, theta: float) -> np.ndarray:
    '''
    In-place (asynchronous) iterative policy evaluation for equiproable policy
    '''
    value_function = np.zeros((env.height, env.width))

    while True:
        delta = 0

        for state in env.state_space:
            if env.terminated(state):
                continue

            x, y = state[0], state[1]
            old_value = value_function[x, y]
            next_history = []

            for action in env.action_space:
                env.state = np.copy(state)
                next_state, reward, _ = env.step(action)
                next_history.append((next_state, action, reward))

            value = 0
            for next_state, action, reward in next_history:
                value += env.transition_probs[action] * 1 * (reward +
                    gamma * value_function[next_state[0], next_state[1]])
            value_function[x, y] = value
            delta = max(delta, abs(old_value - value_function[x, y]))

        if delta < theta:
            break

    return value_function


if __name__ == '__main__':
    height = width = 4
    terminal_states = [(0, 0), (height - 1, width - 1)]
    env = GridWorld(height, width, terminal_states=terminal_states)
    gamma = 1
    theta = 1e-5
    value_function = iterative_policy_evaluation(env, gamma, theta)
    value_function = np.around(np.reshape(
        value_function, (height, width)), decimals=1)
    print(value_function)
