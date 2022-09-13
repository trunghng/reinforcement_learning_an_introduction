import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))

import numpy as np

from env import GridWorld


def get_true_value(env: GridWorld, gamma: float) -> np.ndarray:
    '''
    Compute true value by Bellman equations by
    constructing system of linear equations Ax=b from Bellman equations

    Params
    ------
    env: GridWorld
    gamma: discount factor
    '''
    n_states = len(env.state_space)
    A = np.zeros((n_states, n_states))
    b = np.zeros(n_states)

    for i, state in enumerate(env.state_space):
        next_history = []

        for action in env.action_space:
            env.state = np.copy(state)
            next_state, reward, _ = env.step(action)
            if not (state == next_state).all() and \
                (state[0], state[1]) not in env.states_:
                reward = 0

            next_history.append((next_state, action, reward))

        coefficients = np.zeros(n_states)
        reward_ = 0

        for t, history in enumerate(next_history):
            next_state, action, reward = history
            coefficients[next_state[0] * env.height + next_state[1]] \
                += env.transition_probs[action] * gamma
            reward_ += env.transition_probs[action] * 1 * reward

        coefficients[state[0] * env.height + state[1]] -= 1
        A[i] = coefficients
        b[i] = -reward_

    true_value = np.linalg.solve(A, b)
    return true_value


if __name__ == '__main__':
    height = width = 5
    special_states = [[(0, 1), (0, 3)], [(4, 1), (2, 3)], [10, 5]]
    env = GridWorld(height, width, special_states=special_states)
    gamma = 0.9
    true_value = get_true_value(env, gamma)
    true_value = np.around(np.reshape(
        true_value, (height, width)), decimals=1)
    print(true_value)
