import numpy as np

GRID_SIZE = 5
ACTION_PROB = 0.25
GAMMA = 0.9  # discount factor
ACTIONS = {'north': (-1, 0), 'east': (0, 1), 'south': (1, 0), 'west': (0, -1)}
REWARDS = {'changed': 0, 'unchanged': -1, 'A': 10, 'B': 5}
STATES = {'A': (0, 1), 'B': (0, 3), 'A_prime': (4, 1), 'B_prime': (2, 3)}


def bellman_lin_eqn_sys():
    A = []
    b = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # generate successive state-reward pair for each state
            next_states = []
            next_rewards = []

            if i == STATES['A'][0] and j == STATES['A'][1]:
                for _ in range(4):  # since every action from A is gonna lead to A'
                    next_states.append(STATES['A_prime'])
                    next_rewards.append(REWARDS['A'])
            elif i == STATES['B'][0] and j == STATES['B'][1]:
                for _ in range(4):  # likewise, every action from B is gonna lead to B'
                    next_states.append(STATES['B_prime'])
                    next_rewards.append(REWARDS['B'])
            else:
                for action in ACTIONS:
                    if i + ACTIONS[action][0] == -1 or i + ACTIONS[action][0] == GRID_SIZE or j + ACTIONS[action][1] == -1 or j + ACTIONS[action][1] == GRID_SIZE:
                        next_states.append((i, j))
                        next_rewards.append(REWARDS['unchanged'])
                    else:
                        next_states.append((i + ACTIONS[action][0], j + ACTIONS[action][1]))
                        next_rewards.append(REWARDS['changed'])

            # Construct system of linear equation (coefficient matrix A, RHS b) from the  Bellman equations
            coefficients = [0 for _ in range(GRID_SIZE**2)]
            reward = 0
            for index, s_prime in enumerate(next_states):
                coefficients[s_prime[0] * GRID_SIZE + s_prime[1]] += ACTION_PROB * 1 * GAMMA
                reward += ACTION_PROB * 1 * next_rewards[index]
            coefficients[i * GRID_SIZE + j] -= 1
            A.append(coefficients)
            b.append(-reward)

    return A, b


def get_state_value_function():
    A, b = bellman_lin_eqn_sys()
    return np.linalg.solve(A, b)


if __name__ == '__main__':
    state_value_func = np.array(get_state_value_function())
    state_value_func = np.around(np.reshape(state_value_func, (GRID_SIZE, GRID_SIZE)), decimals=1)
    print(state_value_func)
