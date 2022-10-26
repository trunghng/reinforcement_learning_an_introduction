import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import Gambler

# GOAL = 100
# # For convenience, we introduce 2 dummy states: 0 and terminal state
# states = np.arange(0, GOAL + 1)
# rewards = {'terminal': 1, 'non-terminal': 0}
# HEAD_PROB = 0.4
# GAMMA = 1  # discount factor


# def value_iteration(theta):
#     V = np.zeros(states.shape)
#     V_set = []
#     policy = np.zeros(V.shape)

#     while True:
#         delta = 0
#         V_set.append(V.copy())
#         for state in states[1:GOAL]:

#             old_value = V[state].copy()

#             actions = np.arange(0, min(state, GOAL - state) + 1)
#             new_value = 0
#             for action in actions:
#                 next_head_state = states[state] + action
#                 next_tail_state = states[state] - action
#                 head_reward = rewards['terminal'] if next_head_state == GOAL else rewards['non-terminal']
#                 tail_reward = rewards['non-terminal']
#                 value = HEAD_PROB * (head_reward + GAMMA * V[next_head_state]) + \
#                     (1 - HEAD_PROB) * (tail_reward + GAMMA * V[next_tail_state])
#                 if value > new_value:
#                     new_value = value

#             V[state] = new_value
#             delta = max(delta, abs(old_value - V[state]))
#             print('Max value changed: ', delta)

#         if delta < theta:
#             V_set.append(V)
#             break

#     for state in states[1:GOAL]:
#         values = []
#         actions = np.arange(min(state, 100 - state) + 1)
#         for action in actions:
#             next_head_state = states[state] + action
#             next_tail_state = states[state] - action
#             head_reward = rewards['terminal'] if next_head_state == GOAL else rewards['non-terminal']
#             tail_reward = rewards['non-terminal']
#             values.append(HEAD_PROB * (head_reward + GAMMA * V[next_head_state]) +
#                           (1 - HEAD_PROB) * (tail_reward + GAMMA * V[next_tail_state]))
#         policy[state] = actions[np.argmax(np.round(values[1:], 4)) + 1]

#     return V_set, policy


class ValueIteration:
    '''
    Value iteration
    '''

    def __init__(self, env: Gambler,
            gamma: float, theta: float, head_prob: float) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.head_prob = head_prob


    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        value_function = np.zeros(len(self.env.state_space))
        policy = np.zeros(value_function.shape)
        value_functions = []
        head_prob = self.head_prob

        while True:
            delta = 0
            value_functions.append(value_function.copy())

            for state in self.env.state_space[1:-1]:
                old_value = value_function[state].copy()

                values = []
                for action in self.env.action_space(state):
                    value = 0
                    for head in [True, False]:
                        next_state, reward, _ = self.env.step(state, action, head)
                        value += (head * head_prob + (1 - head) * (1 - head_prob)) \
                            * (reward + self.gamma * value_function[next_state])
                    values.append(value)
                value_function[state] = max(values)
                delta = max(delta, abs(old_value - value_function[state]))
                print('Max value changed: ', delta)

            if delta < self.theta:
                value_functions.append(value_function)
                break

        for state in self.env.state_space[1:-1]:
            values = []
            actions = self.env.action_space(state)
            for action in actions:
                value = 0
                for head in [True, False]:
                    next_state, reward, _ = self.env.step(state, action, head)
                    value +=  (head * head_prob + (1 - head) * (1 - head_prob)) \
                        * (reward + self.gamma * value_function[next_state])
                values.append(value)
            policy[state] = actions[np.argmax(np.round(values[1:], 5)) + 1]

        return value_function, policy


if __name__ == '__main__':
    goal = 100
    head_prob = 0.4
    env = Gambler(goal)
    gamma = 1
    theta = 1e-13
    value_iteration = ValueIteration(env, gamma, theta, head_prob)
    value_functions, optimal_policy = value_iteration.run()
    optimal_value = value_functions[-1]
    print(optimal_value)

    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    for i, value in enumerate(value_functions):
        plt.plot(value, label='sweep {}'.format(i))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.scatter(env.state_space, optimal_policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('./gambler2.png')
    plt.close()
