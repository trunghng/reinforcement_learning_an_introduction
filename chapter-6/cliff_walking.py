import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class GridWorld:


    def __init__(self, height, width, start_state, goal_state, cliff):
        '''
        Initialization function

        Params
        ------
        height: int
            gridworld's height
        width: int
            gridworld's width
        start_state: [int, int]
            gridworld's start state
        goal_state: [int, int]
            gridworld's goal state
        cliff: list<[int, int]>
            gridworld's cliff region
        '''
        self.height = height
        self.width = width
        self.start_state = start_state
        self.goal_state = goal_state
        self.cliff = cliff
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.rewards = {'cliff': -100, 'non-cliff': -1}


    def is_terminal(self, state):
        '''
        Whether state @state is the goal state

        Params
        ------
        state: [int, int]
            current state
        '''
        return state == self.goal_state


    def take_action(self, state, action):
        '''
        Take action @action at state @state

        Params
        ------
        state: [int, int]
            current state
        action: (int, int)
            action taken

        Return
        ------
        (next_state, reward): ([int, int], int)
            a tuple of next state and reward
        '''
        next_state = [state[0] + action[0], state[1] + action[1]]
        next_state = [max(0, next_state[0]), max(0, next_state[1])]
        next_state = [min(self.height - 1, next_state[0]), min(self.width - 1, next_state[1])]
        if next_state in self.cliff:
            reward = self.rewards['cliff']
            next_state = self.start_state
        else:
            reward = self.rewards['non-cliff']
        return next_state, reward


    def get_action_idx(self, action):
        '''
        Get index of action in action list

        Params
        ------
        action: (int, int)
            action
        '''
        return self.actions.index(action)


def epsilon_greedy(grid_world, epsilon, Q, state):
    '''
    Choose action according to epsilon-greedy policy

    Params:
    -------
    grid_world: GridWorld
    epsilon: float
    Q: np.ndarray
        action-value function
    state: [int, int]
        current state

    Return
    ------
    action: (int, int)
    '''
    if np.random.binomial(1, epsilon):
        action_idx = np.random.randint(len(grid_world.actions))
        action = grid_world.actions[action_idx]
    else:
        values = Q[state[0], state[1], :]
        action_idx = np.random.choice([action_ for action_, value_ 
            in enumerate(values) if value_ == np.max(values)])
        action = grid_world.actions[action_idx]
    return action


def q_learning(Q, grid_world, epsilon, alpha, gamma):
    '''
    Q-learning

    Params
    ------
    Q: np.ndarray
        action-value function
    grid_world: GridWorld
    epsilon: float
    alpha: float
        step size
    gamma: float
        discount factor
    '''
    state = grid_world.start_state
    rewards = 0

    while not grid_world.is_terminal(state):
        action = epsilon_greedy(grid_world, epsilon, Q, state)
        next_state, reward = grid_world.take_action(state, action)
        rewards += reward
        action_idx = grid_world.get_action_idx(action)
        Q[state[0], state[1], action_idx] += alpha * (reward + gamma * \
            np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action_idx])
        state = next_state

    return rewards


def sarsa(Q, grid_world, epsilon, alpha, gamma):
    '''
    Sarsa

    Params
    ------
    Q: np.ndarray
        action-value function
    grid_world: GridWorld
    epsilon: float
    alpha: float
        step size
    gamma: float
        discount factor
    '''
    state = grid_world.start_state
    action = epsilon_greedy(grid_world, epsilon, Q, state)
    rewards = 0

    while not grid_world.is_terminal(state):
        next_state, reward = grid_world.take_action(state, action)
        rewards += reward
        next_action = epsilon_greedy(grid_world, epsilon, Q, next_state)
        action_idx = grid_world.get_action_idx(action)
        next_action_idx = grid_world.get_action_idx(next_action)
        Q[state[0], state[1], action_idx] += alpha * (reward + gamma * Q[next_state[0], \
            next_state[1], next_action_idx] - Q[state[0], state[1], action_idx])
        state = next_state
        action = next_action

    return rewards


def print_optimal_policy(Q, grid_world):
    for i in range(grid_world.height):
        optimal_policy_row = []
        for j in range(grid_world.width):
            if grid_world.is_terminal([i, j]):
                optimal_policy_row.append('G')
                continue
            best_action_idx = np.argmax(Q[i, j, :])
            if best_action_idx == 0:
                optimal_policy_row.append('U')
            elif best_action_idx == 1:
                optimal_policy_row.append('D')
            elif best_action_idx == 2:
                optimal_policy_row.append('R')
            elif best_action_idx == 3:
                optimal_policy_row.append('L')
        print(optimal_policy_row)


if __name__ == '__main__':
    height = 4
    width = 13
    start_state = [3, 0]
    goal_state = [3, 12]
    cliff = [[3, x] for x in range(1, 12)]
    grid_world = GridWorld(height, width, start_state, goal_state, cliff)
    n_runs = 50
    n_eps = 500
    epsilon = 0.1
    alpha = 0.5
    gamma = 1
    Q = np.zeros((height, width, len(grid_world.actions)))
    rewards_q_learning = np.zeros(n_eps)
    rewards_sarsa = np.zeros(n_eps)

    for _ in tqdm(range(n_runs)):
        Q_q_learning = Q.copy()
        Q_sarsa = Q.copy()

        for ep in range(n_eps):
            rewards_q_learning[ep] += q_learning(Q_q_learning, grid_world, epsilon, alpha, gamma)
            rewards_sarsa[ep] += sarsa(Q_sarsa, grid_world, epsilon, alpha, gamma)

    rewards_q_learning /= n_runs
    rewards_sarsa /= n_runs

    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('./cliff_walking.png')
    plt.close()

    print('Q-learning\'s optimal policy:')
    print_optimal_policy(Q_q_learning, grid_world)
    print('Sarsa\'s optimal policy:')
    print_optimal_policy(Q_sarsa, grid_world)
