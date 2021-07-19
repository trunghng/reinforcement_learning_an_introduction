import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


MAZE_HEIGHT = 6
MAZE_WIDTH = 9
TERMINAL_STATE = [0, 8]
START_STATE = [2, 0]
OBSTACLES = [[0, 7], [1, 7], [2, 7], [1, 2], [2, 2], [3, 2], [4, 5]]
ACTIONS = {'up': (-1, 0), 'down':(1, 0), 'right': (0, 1), 'left': (0, -1)}
ACTION_NAMES = list(ACTIONS.keys())
REWARDS = {'goal': 1, 'non-goal': 0}


class Model:
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand


    def learning(self, state, action, next_state, reward):
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = next_state, reward


    def search(self):
        states = list(self.model.keys())
        state = states[self.rand.choice(len(states))]
        actions = list(self.model[tuple(state)].keys())
        action = actions[self.rand.choice(len(actions))]
        next_state, reward = self.model[tuple(state)][action]
        return state, action, next_state, reward


def is_terminal(state):
    return state == TERMINAL_STATE


def take_action(state, action):
    next_state = [state[0] + action[0], state[1] + action[1]]
    next_state = [max(0, next_state[0]), max(0, next_state[1])]
    next_state = [min(MAZE_HEIGHT - 1, next_state[0]), min(MAZE_WIDTH - 1, next_state[1])]
    if is_terminal(next_state):
        reward = REWARDS['goal']
    else:
        reward = REWARDS['non-goal']
        if next_state in OBSTACLES:
            next_state = state
    return next_state, reward


def epsilon_greedy(Q, epsilon, state):
    if np.random.binomial(1, epsilon) == 1:
        action = ACTION_NAMES.index(np.random.choice(ACTION_NAMES))
    else:
        values = Q[state[0], state[1]]
        action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])
    return action


def dyna_q(Q, model, epsilon, alpha, gamma, planning_step):
    state = START_STATE
    step = 0

    while not is_terminal(state):
        action = epsilon_greedy(Q, epsilon, state)
        action_name = ACTION_NAMES[action]
        next_state, reward = take_action(state, ACTIONS[action_name])
        Q[state[0], state[1], action] += alpha * (reward + gamma * 
            np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])
        model.learning(state, action, next_state, reward)
        for _ in range(planning_step):
            state_, action_, next_state_, reward_ = model.search()
            Q[state_[0], state_[1], action_] += alpha * (reward_ + gamma * 
                np.max(Q[next_state_[0], next_state_[1], :]) - Q[state_[0], state_[1], action_])
        state = next_state
        step += 1
    return step


def dyna_maze():
    runs = 30
    episodes = 50 
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))

    for _ in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            Q = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, len(ACTION_NAMES)))
            model = Model()

            for ep in range(episodes):
                steps[i, ep] += dyna_q(Q, model, epsilon, alpha, gamma, planning_step)

    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label=f'{planning_steps[i]} planning steps')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend()

    plt.savefig('./dyna_maze.png')
    plt.close()


if __name__ == '__main__':
    dyna_maze()










