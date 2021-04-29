import numpy as np

GRID_SIZE = 4
actions = {'north': (-1, 0), 'east': (0, 1), 'south': (1, 0), 'west': (0, -1)}
reward = -1
ACTION_PROB = 0.25
GAMMA = 1


def is_terminal(state):
    return state == (0, 0) or state == (GRID_SIZE - 1, GRID_SIZE - 1)


def iterative_policy_evaluation(theta):
    """
    In-place (asynchronous) iterative policy evaluation for equiproable policy
    """
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    while True:
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if is_terminal((i, j)):
                    continue

                old_value = V[i, j]

                next_states = []
                for action in actions:
                    if i + actions[action][0] == GRID_SIZE or i + actions[action][0] == -1 or j + actions[action][1] == GRID_SIZE or j + actions[action][1] == -1:
                        next_states.append((i, j))
                    else:
                        next_states.append((i + actions[action][0], j + actions[action][1]))
                value = 0
                for state in next_states:
                    value += ACTION_PROB * 1 * (reward + GAMMA * V[state[0], state[1]])
                V[i, j] = value

                delta = max(delta, abs(old_value - V[i, j]))

        if delta < theta:
            break

    return V


if __name__ == '__main__':
    theta = 1e-5
    state_value_func = iterative_policy_evaluation(theta)
    state_value_func = np.around(np.reshape(state_value_func, (GRID_SIZE, GRID_SIZE)), decimals=1)
    print(state_value_func)
