import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

ACTIONS = {'up': (-1, 0), 'down':(1, 0), 'right': (0, 1), 'left': (0, -1)}
ACTION_NAMES = list(ACTIONS.keys())
GRID_HEIGHT = 7
GRID_WIDTH = 10
WIND_DIST = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
TERMINAL_STATE = [3, 7]
START_STATE = [3, 0]
REWARDS = -1

def take_action(state, action):
    next_state = [state[0] + action[0] - WIND_DIST[state[1]], state[1] + action[1]]
    next_state = [max(0, next_state[0]), max(0, next_state[1])]
    next_state = [min(GRID_HEIGHT - 1, next_state[0]), min(GRID_WIDTH - 1, next_state[1])]
    return next_state


def is_terminal(state):
    return state == TERMINAL_STATE


def epsilon_greedy(epsilon, action_values, state):
    if np.random.binomial(1, epsilon):
        action = ACTION_NAMES.index(np.random.choice(ACTION_NAMES))
    else:
        values = action_values[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])
    return action


def sarsa(episodes, alpha, epsilon, gamma):
    time_steps = []
    Q = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTION_NAMES)))

    for _ in tqdm(range(episodes)):
        steps = 0
        state = START_STATE
        action = epsilon_greedy(epsilon, Q, state)

        while not is_terminal(state):
            steps += 1
            action_name = ACTION_NAMES[action]
            next_state = take_action(state, ACTIONS[action_name])
            next_action = epsilon_greedy(epsilon, Q, next_state)
            Q[state[0], state[1], action] += alpha * (REWARDS + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
            state = next_state
            action = next_action
        time_steps.append(steps)

    return Q, time_steps


def print_optimal_policy(Q):
    print('Optimal policy:')
    for i in range(GRID_HEIGHT):
        optimal_policy_row = []
        for j in range(GRID_WIDTH):
            if is_terminal([i, j]):
                optimal_policy_row.append('G')
                continue
            best_action = np.argmax(Q[i, j, :])
            if ACTION_NAMES[best_action] == 'up':
                optimal_policy_row.append('U')
            elif ACTION_NAMES[best_action] == 'down':
                optimal_policy_row.append('D')
            elif ACTION_NAMES[best_action] == 'left':
                optimal_policy_row.append('L')
            elif ACTION_NAMES[best_action] == 'right':
                optimal_policy_row.append('R')
        print(optimal_policy_row)


if __name__ == '__main__':
    episodes = 600
    epsilon = 0.1
    alpha = 0.5
    gamma = 1

    Q, time_steps = sarsa(episodes, alpha, epsilon, gamma)
    time_steps = np.add.accumulate(time_steps)

    plt.plot(time_steps, np.arange(1, len(time_steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./windy_gridworld.png')
    plt.close()

    print_optimal_policy(Q)

    
