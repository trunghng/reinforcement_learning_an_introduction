import numpy as np
import matplotlib.pyplot as plt

GOAL = 100
# For convenience, we introduce 2 dummy states: 0 and terminal state
states = np.arange(0, GOAL + 1)
rewards = {'terminal': 1, 'non-terminal': 0}
HEAD_PROB = 0.4
GAMMA = 1  # discount factor


def value_iteration(theta):
    V = np.zeros(states.shape)
    V_set = []
    policy = np.zeros(V.shape)

    while True:
        delta = 0
        V_set.append(V.copy())
        for state in states[1:GOAL]:

            old_value = V[state].copy()

            actions = np.arange(0, min(state, GOAL - state) + 1)
            new_value = 0
            for action in actions:
                next_head_state = states[state] + action
                next_tail_state = states[state] - action
                head_reward = rewards['terminal'] if next_head_state == GOAL else rewards['non-terminal']
                tail_reward = rewards['non-terminal']
                value = HEAD_PROB * (head_reward + GAMMA * V[next_head_state]) + \
                    (1 - HEAD_PROB) * (tail_reward + GAMMA * V[next_tail_state])
                if value > new_value:
                    new_value = value

            V[state] = new_value
            delta = max(delta, abs(old_value - V[state]))
            print('Max value changed: ', delta)

        if delta < theta:
            V_set.append(V)
            break

    for state in states[1:GOAL]:
        values = []
        actions = np.arange(min(state, 100 - state) + 1)
        for action in actions:
            next_head_state = states[state] + action
            next_tail_state = states[state] - action
            head_reward = rewards['terminal'] if next_head_state == GOAL else rewards['non-terminal']
            tail_reward = rewards['non-terminal']
            values.append(HEAD_PROB * (head_reward + GAMMA * V[next_head_state]) +
                          (1 - HEAD_PROB) * (tail_reward + GAMMA * V[next_tail_state]))
        policy[state] = actions[np.argmax(np.round(values[1:], 4)) + 1]

    return V_set, policy


if __name__ == '__main__':
    theta = 1e-13
    value_funcs, optimal_policy = value_iteration(theta)
    optimal_value = value_funcs[-1]
    print(optimal_value)

    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    for sweep, value in enumerate(value_funcs):
        plt.plot(value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.scatter(states, optimal_policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('./gambler.png')
    plt.close()
