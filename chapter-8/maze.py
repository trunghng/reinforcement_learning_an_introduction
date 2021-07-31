import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import heapq
import itertools


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


class TimeBasedModel(Model):

    def __init__(self, maze, kappa, rand=np.random):
        Model.__init__(self, rand)
        self.maze = maze
        #time weight
        self.kappa = kappa
        self.current_time_step = 0


    def model_learning(self, state, action, next_state, reward):
        self.current_time_step += 1
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
            for action_ in range(len(self.maze.actions)):
                if action_ != action:
                    # lead back to the same state with a reward of zero
                    self.model[tuple(state)][action_] = state, 0, 1
        self.model[tuple(state)][action] = next_state, reward, self.current_time_step


    def search_control(self):
        states = list(self.model.keys())
        state = states[self.rand.choice(len(states))]
        actions = list(self.model[tuple(state)].keys())
        action = actions[self.rand.choice(len(actions))]
        next_state, reward, last_time_tried = self.model[tuple(state)][action]
        reward += self.kappa * np.sqrt(self.current_time_step - last_time_tried)
        return state, action, next_state, reward


class PQueueModel(Model):

    def __init__(self, rand=np.random):
        Model.__init__(self, rand)
        self.pqueue = PriorityQueue()
        self.predecessor_pairs = dict()


    def model_learning(self, state, action, next_state, reward):
        Model.model_learning(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessor_pairs.keys():
            self.predecessor_pairs[tuple(next_state)] = set()
        self.predecessor_pairs[tuple(next_state)].add((tuple(state), action))


    def search_control(self):
        state, action = self.pqueue.pop()
        next_state, reward = self.model[tuple(state)][action]
        return state, action, next_state, reward


    # return state, action, reward predicted to lead to a state
    def get_predecessor_pairs(self, state):
        predecessors_ = []
        if tuple(state) in self.predecessor_pairs.keys():
            for pre_state, pre_action in list(self.predecessor_pairs[tuple(state)]):
                predecessors_.append([pre_state, pre_action, 
                    self.model[tuple(pre_state)][pre_action][1]])
        return predecessors_


class Maze:

    def __init__(self, 
                height, 
                width, 
                start_state, 
                terminal_states, 
                obstacles, 
                max_step=float('inf'), 
                obstacles_change_step=None):
        self._height = height
        self._width = width
        self._start_state = start_state
        self._terminal_states = terminal_states
        self._obstacles = obstacles
        self._actions = {'up': (-1, 0), 'down':(1, 0), 'right': (0, 1), 'left': (0, -1)}
        self._action_names = list(self._actions.keys())
        self._rewards = {'goal': 1, 'non-goal': 0}
        self._max_step = max_step
        self._obstacles_change_step = obstacles_change_step


    @property
    def height(self):
        return self._height
    

    @property
    def width(self):
        return self._width
    

    @property
    def start_state(self):
        return self._start_state
    
    
    @property
    def terminal_states(self):
        return self._terminal_states


    @property
    def obstacles(self):
        return self._obstacles


    @obstacles.setter
    def obstacles(self, obstacles):
        self._obstacles = obstacles


    @property
    def actions(self):
        return self._actions


    @property
    def action_names(self):
        return self._action_names


    @property
    def rewards(self):
        return self._rewards


    @property
    def max_step(self):
        return self._max_step
    

    @property
    def obstacles_change_step(self):
        return self._obstacles_change_step
    

    def is_terminal(self, state):
        return state in self._terminal_states


    def __repr__ (self):
        return f'{self.start_state}, {self.terminal_states}, {self.obstacles}'


    def take_action(self, state, action):
        next_state = [state[0] + action[0], state[1] + action[1]]
        next_state = [max(0, next_state[0]), max(0, next_state[1])]
        next_state = [min(self.height - 1, next_state[0]), min(self.width - 1, next_state[1])]
        if self.is_terminal(next_state):
            reward = self._rewards['goal']
        else:
            reward = self._rewards['non-goal']
            if next_state in self._obstacles:
                next_state = state
        return next_state, reward


    def extend_state(self, state, resolution):
        new_states = []
        for i in range(state[0] * resolution, (state[0] + 1) * resolution):
            for j in range(state[1] * resolution, (state[1] + 1) * resolution):
                new_states.append([i, j])
        return new_states


    def extend(self, resolution):
        height_ = self._height * resolution
        width_ = self._width * resolution
        start_state_ = [self._start_state[0] * resolution, self._start_state[1] * resolution]
        terminal_states_ = [_ for _ in self.extend_state(self._terminal_states[0], resolution)]
        obstacles_ = []
        for obstacle_ in self._obstacles:
            obstacle_.extend(self.extend_state(obstacle_, resolution))
        maze_ = Maze(height_, width_, start_state_, terminal_states_, obstacles_)
        return maze_


class PriorityQueue:
    '''
    This class is taken and modified from 
    https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes 
    '''

    def __init__(self):
        self.pqueue = []
        self.entry_finder = {}
        self.REMOVED = '<removed-item>'
        self.counter = itertools.count()


    def push(self, item, priority=0):
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pqueue, entry)


    def remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1]  = self.REMOVED


    def pop(self):
        while self.pqueue:
            priority, count, item = heapq.heappop(self.pqueue)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')


    def is_empty(self):
        return not self.entry_finder


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


def prioritized_sweeping(maze, Q, model, epsilon, alpha, gamma, planning_step, theta):
    state = maze.start_state
    step = 0
    updates = 0

    while not maze.is_terminal(state):
        action = epsilon_greedy(maze, Q, epsilon, state)
        action_name = maze.action_names[action]
        next_state, reward = maze.take_action(state, maze.actions[action_name])

        model.model_learning(state, action, next_state, reward)

        priority = np.abs(reward + gamma * np.max(Q[next_state[0], next_state[1], :]) 
            - Q[state[0], state[1], action])
        if priority > theta:
            # since the heap in heapq is a min-heap
            model.pqueue.push((tuple(state), action), -priority)

        planning_step_count = 0
        while planning_step_count < planning_step and not model.pqueue.is_empty():
            state_, action_, next_state_, reward_ = model.search_control()
            Q[state_[0], state_[1], action_] += alpha * (reward_ + gamma * 
                np.max(Q[next_state_[0], next_state_[1], :]) - Q[state_[0], state_[1], action_])
            for pre_state_, pre_action_, pre_reward_ in model.get_predecessor_pairs(state_):
                priority_ = np.abs(pre_reward_ + gamma * np.max(Q[state_[0], state_[1], :]) 
                    - Q[pre_state_[0], pre_state_[1], pre_action_])
                if priority_ > theta:
                    model.pqueue.push((tuple(pre_state_), pre_action_), -priority_)
            planning_step_count += 1

        state = next_state
        step += 1
        updates += planning_step_count + 1

    return updates


def dyna_maze():
    height = 6
    width = 9
    start_state = [2, 0]
    terminal_states = [[0, 8]]
    obstacles = [[0, 7], [1, 7], [2, 7], [1, 2], [2, 2], [3, 2], [4, 5]]
    maze = Maze(height, width, start_state, terminal_states, obstacles)

    runs = 30
    episodes = 50 
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))

    for _ in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            Q = np.zeros((maze.height, maze.width, len(maze.action_names)))
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
    height = 6
    width = 9
    start_state = [5, 3]
    terminal_states = [[0, 8]]
    obstacles = [[3, i] for i in range(8)]
    new_obstacles = [[3, i] for i in range(1, 9)]

    runs = 20
    alpha = 1
    epsilon = 0.1
    gamma = 0.95
    planning_step = 10
    obstacles_change_step = 1000
    max_step = 3000
    kappa = 1e-4
    methods = ['Dyna-Q', 'Dyna-Q+']
    rewards = np.zeros((runs, len(methods), max_step))
    maze = Maze(height, width, start_state, terminal_states, obstacles, max_step, obstacles_change_step)
    
    
    for i, method in enumerate(methods):
        print(method)

        for run in tqdm(range(runs)):
            Q = np.zeros((maze.height, maze.width, len(maze.action_names)))
            if method == 'Dyna-Q':
                model = Model()
            else:
                model = TimeBasedModel(maze, kappa)

            maze.obstacles = obstacles

            step = 0
            last_step = 0
            while step < maze.max_step:
                step += dyna_q(maze, Q, model, epsilon, alpha, gamma, planning_step)
    
                rewards[run, i, last_step: step] = rewards[run, i, last_step]
                rewards[run, i, min(step, maze.max_step - 1)] = rewards[run, i, last_step] + 1
                last_step = step

                if step > maze.obstacles_change_step:
                    maze.obstacles = new_obstacles
        time.sleep(0.1)

    rewards = np.mean(rewards, axis=0)

    for i in range(len(methods)):
        plt.plot(rewards[i, :], label=methods[i])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.savefig('./blocking_maze.png')
    plt.close()


def shortcut_maze():
    height = 6
    width = 9
    start_state = [5, 3]
    terminal_states = [[0, 8]]
    obstacles = [[3, i] for i in range(1, 9)]
    new_obstacles = [[3, i] for i in range(1, 8)]

    runs = 5
    alpha = 1
    epsilon = 0.1
    gamma = 0.95
    planning_step = 50
    obstacles_change_step = 3000
    max_step = 6000
    kappa = 1e-3
    methods = ['Dyna-Q', 'Dyna-Q+']
    rewards = np.zeros((runs, len(methods), max_step))
    maze = Maze(height, width, start_state, terminal_states, obstacles, max_step, obstacles_change_step)

    for i, method in enumerate(methods):
        print(method)

        for run in tqdm(range(runs)):
            Q = np.zeros((maze.height, maze.width, len(maze.action_names)))
            if method == 'Dyna-Q':
                model = Model()
            else:
                model = TimeBasedModel(maze, kappa)

            maze.obstacles = obstacles

            step = 0
            last_step = 0
            while step < maze.max_step:
                step += dyna_q(maze, Q, model, epsilon, alpha, gamma, planning_step)
    
                rewards[run, i, last_step: step] = rewards[run, i, last_step]
                rewards[run, i, min(step, maze.max_step - 1)] = rewards[run, i, last_step] + 1
                last_step = step

                if step > maze.obstacles_change_step:
                    maze.obstacles = new_obstacles
        time.sleep(0.1)

    rewards = np.mean(rewards, axis=0)

    for i in range(len(methods)):
        plt.plot(rewards[i, :], label=methods[i])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.savefig('./shortcut_maze.png')
    plt.close()


def is_optimal_solution(Q, maze, resolution):
    max_steps = 14 * resolution * 1.2
    state = maze.start_state
    steps = 0
    while not maze.is_terminal(state):
        action = np.argmax(Q[state[0], state[1], :])
        action_name = maze.action_names[action]
        state, _ = maze.take_action(state, maze.actions[action_name])
        steps += 1
        if steps > max_steps:
            return False
    return True


def mazes():
    height = 6
    width = 9
    start_state = [2, 0]
    terminal_states = [[0, 8]]
    obstacles = [[0, 7], [1, 7], [2, 7], [1, 2], [2, 2], [3, 2], [4, 5]]
    maze = Maze(height, width, start_state, terminal_states, obstacles)
    resolutions = range(1, 6)

    runs = 5
    alpha = 0.5
    epsilon = 0.1
    gamma = 0.95
    planning_step = 5
    theta = 0.0001
    methods = {
        'Dyna-Q': {
            'model': Model,
            'method': dyna_q,
            'params': [epsilon, alpha, gamma, planning_step]
        },
        'Prioritized Sweeping': {
            'model': PQueueModel,
            'method': prioritized_sweeping,
            'params': [epsilon, alpha, gamma, planning_step, theta]
        }
    }

    updates = np.zeros((runs, len(methods), len(resolutions)))

    for i, (method_name, method) in enumerate(methods.items()):
        print(method_name)

        for run in tqdm(range(runs)):
            print()
            for res in resolutions:
                maze_ = maze.extend(res)
                print(f'run = {run}, maze size = {maze_.height * maze_.width}')

                Q = np.zeros((maze_.height, maze_.width, len(maze_.action_names)))
                model = method['model']()
                steps = []

                while True:
                    steps.append(method['method'](maze_, Q, model, *method['params']))

                    if is_optimal_solution(Q, maze_, res):
                        break

                updates[run, i, res - 1] = np.sum(steps)
        time.sleep(0.1)

    updates = np.mean(updates, axis=0)

    updates[0, :] *= planning_step + 1

    for i, method_name in enumerate(methods):
        plt.plot(np.arange(1, len(resolutions) + 1), updates[i, :], label=method_name)
    plt.xlabel('maze resolution factor')
    plt.ylabel('updates until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('./prioritized_sweeping.png')
    plt.close()


if __name__ == '__main__':
    dyna_maze()
    blocking_maze()
    shortcut_maze()
    mazes()
