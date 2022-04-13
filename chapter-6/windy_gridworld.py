import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class WindyGridWorld:


    def __init__(self, height, width, start_state, end_state, wind_dist):
        self.height = height
        self.width = width
        self.start_state = start_state
        self.end_state = end_state
        self.wind_dist = wind_dist
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.reward = -1


    def is_terminal(self, state):
        '''
        Whether state @state is an end state

        Params
        ------
        state: [int, int]
            current state
        '''
        return state == self.end_state


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
        next_state = [state[0] + action[0] - self.wind_dist[state[1]], state[1] + action[1]]
        next_state = [max(0, next_state[0]), max(0, next_state[1])]
        next_state = [min(self.height - 1, next_state[0]), min(self.width - 1, next_state[1])]
        reward = self.reward
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


def sarsa(grid_world, n_eps, alpha, epsilon, gamma):
    '''
    Sarsa

    Params
    ------
    grid_world: GridWorld
    n_eps: int
        number of episodes
    alpha: float
        step size
    epsilon: float
    gamma: float
        discount factor
    '''
    time_steps = []
    Q = np.zeros((grid_world.height, grid_world.width, len(grid_world.actions)))

    for _ in tqdm(range(n_eps)):
        steps = 0
        state = grid_world.start_state
        action = epsilon_greedy(grid_world ,epsilon, Q, state)

        while not grid_world.is_terminal(state):
            steps += 1
            next_state, reward = grid_world.take_action(state, action)
            next_action = epsilon_greedy(grid_world, epsilon, Q, next_state)
            action_idx = grid_world.get_action_idx(action)
            next_action_idx = grid_world.get_action_idx(next_action)
            Q[state[0], state[1], action_idx] += alpha * (reward + gamma * 
                Q[next_state[0], next_state[1], next_action_idx] - Q[state[0], state[1], action_idx])
            state = next_state
            action = next_action
        time_steps.append(steps)

    return Q, time_steps


def print_optimal_policy(Q, grid_world):
    print('Optimal policy:')
    for i in range(grid_world.height):
        optimal_policy_row = []
        for j in range(grid_world.width):
            if grid_world.is_terminal([i, j]):
                optimal_policy_row.append('E')
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
    height = 7
    width = 10
    wind_dist = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    start_state = [3, 0]
    end_state = [3, 7]
    grid_world = WindyGridWorld(height, width, start_state, end_state, wind_dist)
    n_eps = 600
    epsilon = 0.1
    alpha = 0.5
    gamma = 1

    Q, time_steps = sarsa(grid_world, n_eps, alpha, epsilon, gamma)
    time_steps = np.add.accumulate(time_steps)

    plt.plot(time_steps, np.arange(1, len(time_steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./windy_gridworld.png')
    plt.close()

    print_optimal_policy(Q, grid_world)

    
