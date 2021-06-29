import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


GRID_HEIGHT = 4
GRID_WIDTH = 13
TERMINAL_STATE = [3, 12]
START_STATE = [3, 0]
CLIFF = [[3, x] for x in range(1, 12)]
ACTIONS = {'up': (-1, 0), 'down': (1, 0), 'right': (0, 1), 'left': (0, -1)}
ACTION_NAMES = list(ACTIONS.keys())
REWARDS = {'cliff': -100, 'non-cliff': -1}


def is_terminal(state):
    return state == TERMINAL_STATE


def take_action(state, action):
    next_state = [state[0] + action[0], state[1] + action[1]]
    next_state = [max(0, next_state[0]), max(0, next_state[1])]
    next_state = [min(GRID_HEIGHT - 1, next_state[0]), min(GRID_WIDTH - 1, next_state[1])]
    if next_state in CLIFF:
        reward = REWARDS['cliff']
        next_state = START_STATE
    else:
        reward = REWARDS['non-cliff']
    return next_state, reward


def epsilon_greedy(epsilon, Q, state):
    if np.random.binomial(1, epsilon) == 1:
        action = ACTION_NAMES.index(np.random.choice(ACTION_NAMES))
    else:
        values = Q[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])
    return action


def q_learning(Q, epsilon, alpha, gamma):
    state = START_STATE
    rewards = 0

    while not is_terminal(state):
        action = epsilon_greedy(epsilon, Q, state)
        action_name = ACTION_NAMES[action]
        next_state, reward = take_action(state, ACTIONS[action_name])
        rewards += reward
        Q[state[0], state[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])
        state = next_state

    return rewards


def sarsa(Q, epsilon, alpha, gamma):
    state = START_STATE
    action = epsilon_greedy(epsilon, Q, state)
    rewards = 0

    while not is_terminal(state):
        action_name = ACTION_NAMES[action]
        next_state, reward = take_action(state, ACTIONS[action_name])
        rewards += reward
        next_action = epsilon_greedy(epsilon, Q, next_state)
        Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], next_action]
            - Q[state[0], state[1], action])
        state = next_state
        action = next_action

    return rewards


def print_optimal_policy(Q):
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
    runs = 50
    episodes = 500
    epsilon = 0.1
    alpha = 0.5
    gamma = 1
    Q = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTION_NAMES)))
    rewards_q_learning = np.zeros((episodes, ))
    rewards_sarsa = np.zeros((episodes, ))

    for _ in tqdm(range(runs)):
        Q_q_learning = Q.copy()
        Q_sarsa = Q.copy()

        for ep in range(episodes):
            rewards_q_learning[ep] += q_learning(Q_q_learning, epsilon, alpha, gamma)
            rewards_sarsa[ep] += sarsa(Q_sarsa, epsilon, alpha, gamma)

    rewards_q_learning /= runs
    rewards_sarsa /= runs

    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('./cliff_walking.png')
    plt.close()

    print('Q-learning\'s optimal policy:')
    print_optimal_policy(Q_q_learning)
    print('Sarsa\'s optimal policy:')
    print_optimal_policy(Q_sarsa)




