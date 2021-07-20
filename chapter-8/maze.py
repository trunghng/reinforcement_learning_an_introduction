import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Model:
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand


    def model_learning(self, state, action, next_state, reward):
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = next_state, reward


    def search_control(self):
        states = list(self.model.keys())
        state = states[self.rand.choice(len(states))]
        actions = list(self.model[tuple(state)].keys())
        action = actions[self.rand.choice(len(actions))]
        next_state, reward = self.model[tuple(state)][action]
        return state, action, next_state, reward


class Maze:
    def __init__(self, start_state, terminal_state, obstacles):
        self.MAZE_HEIGHT = 6
        self.MAZE_WIDTH = 9
        self._start_state = start_state
        self._terminal_state = terminal_state
        self._obstacles = obstacles
        self._actions = {'up': (-1, 0), 'down':(1, 0), 'right': (0, 1), 'left': (0, -1)}
        self._action_names = list(self._actions.keys())
        self._rewards = {'goal': 1, 'non-goal': 0}


    @property
    def start_state(self):
        return self._start_state
    
    
    @property
    def terminal_state(self):
        return self._terminal_state


    @property
    def actions(self):
        return self._actions


    @property
    def action_names(self):
        return self._action_names
    

    def is_terminal(self, state):
        return state == self._terminal_state


    def take_action(self, state, action):
        next_state = [state[0] + action[0], state[1] + action[1]]
        next_state = [max(0, next_state[0]), max(0, next_state[1])]
        next_state = [min(self.MAZE_HEIGHT - 1, next_state[0]), min(self.MAZE_WIDTH - 1, next_state[1])]
        if self.is_terminal(next_state):
            reward = self._rewards['goal']
        else:
            reward = self._rewards['non-goal']
            if next_state in self._obstacles:
                next_state = state
        return next_state, reward


def epsilon_greedy(maze, Q, epsilon, state):
    if np.random.binomial(1, epsilon) == 1:
        action = maze.action_names.index(np.random.choice(maze.action_names))
    else:
        values = Q[state[0], state[1]]
        action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])
    return action


def dyna_q(maze, Q, model, epsilon, alpha, gamma, planning_step):
    state = maze.start_state
    step = 0

    while not maze.is_terminal(state):
        action = epsilon_greedy(maze, Q, epsilon, state)
        action_name = maze.action_names[action]
        next_state, reward = maze.take_action(state, maze.actions[action_name])
        Q[state[0], state[1], action] += alpha * (reward + gamma * 
            np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])
        model.model_learning(state, action, next_state, reward)
        for _ in range(planning_step):
            state_, action_, next_state_, reward_ = model.search_control()
            Q[state_[0], state_[1], action_] += alpha * (reward_ + gamma * 
                np.max(Q[next_state_[0], next_state_[1], :]) - Q[state_[0], state_[1], action_])
        state = next_state
        step += 1
    return step


def dyna_maze():
    terminal_state = [0, 8]
    start_state = [2, 0]
    obstacles = [[0, 7], [1, 7], [2, 7], [1, 2], [2, 2], [3, 2], [4, 5]]
    maze = Maze(start_state, terminal_state, obstacles)

    runs = 30
    episodes = 50 
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))

    for _ in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            Q = np.zeros((maze.MAZE_HEIGHT, maze.MAZE_WIDTH, len(maze.action_names)))
            model = Model()

            for ep in range(episodes):
                steps[i, ep] += dyna_q(maze, Q, model, epsilon, alpha, gamma, planning_step)

    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label=f'{planning_steps[i]} planning steps')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend()

    plt.savefig('./dyna_maze.png')
    plt.close()


def blocking_maze():
    pass


if __name__ == '__main__':
    dyna_maze()










