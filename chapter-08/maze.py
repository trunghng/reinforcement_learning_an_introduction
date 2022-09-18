import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from typing import Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time
import heapq
import itertools

from env import GridWorld


class PriorityQueue:
    '''
    This class is taken and modified from 
    https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes 
    '''

    def __init__(self) -> None:
        self.pqueue = []
        self.entry_finder = {}
        self.REMOVED = '<removed-item>'
        self.counter = itertools.count()


    def push(self, item: object, priority: float=0) -> None:
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pqueue, entry)


    def remove(self, item: object) -> None:
        entry = self.entry_finder.pop(item)
        entry[-1]  = self.REMOVED


    def pop(self) -> object:
        while self.pqueue:
            priority, count, item = heapq.heappop(self.pqueue)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')


    def is_empty(self) -> bool:
        return not self.entry_finder


class DynaAgent(ABC):
    '''
    Dyna agent abstract class
    '''

    def __init__(self, env: GridWorld,
                epsilon: float,
                alpha: float,
                gamma: float,
                planning_step: int) -> None:
        '''
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        planning_step: number of planning steps
        '''
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.planning_step = planning_step
        self.value_function = np.zeros((env.height,
            env.width, len(env.action_space)))
        self.model = dict()


    @abstractmethod
    def __call__(self, env: GridWorld,
                epsilon: float,
                alpha: float, 
                gamma: float,
                planning_step: int) -> object:
        pass


    def reset(self):
        return self.env.reset()


    def _epsilon_greedy(self, state: np.ndarray) -> int:
        '''
        Choose action according to epsilon-greedy

        Params
        ------
        state: state of the agent

        Return
        ------
        action: chosen action
        '''
        if np.random.binomial(1, self.epsilon) == 1:
            action = np.random.choice(self.env.action_space)
        else:
            max_value = self.value_function[state[0], state[1], :].max()
            action = np.random.choice(np.flatnonzero(
                self.value_function[state[0], state[1], :] == max_value))
        return action


    def _update_Q(self, state: np.ndarray,
                action: int, next_state: np.ndarray,
                reward: float) -> None:
        '''
        Update state-action value function

        Params
        ------
        state: state of the agent
        action: action taken at @state
        next_state: next state according to @state
        reward: reward taken at @next_state
        '''
        target = reward + self.gamma * np.max(
            self.value_function[next_state[0], next_state[1], :])
        error = target - self.value_function[state[0], state[1], action]
        self.value_function[state[0], state[1], action] += self.alpha * error


    @abstractmethod
    def _model_learning(self) -> None:
        pass


    @abstractmethod
    def _search_control(self) -> None:
        pass


    @abstractmethod
    def run(self) -> None:
        pass


class DynaQ(DynaAgent):
    '''
    Dyna-Q agent
    '''

    def __init__(self, env: GridWorld,
                epsilon: float,
                alpha: float,
                gamma: float,
                planning_step: int) -> None:
        '''
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        planning_step:
        '''
        super().__init__(env, epsilon, alpha, gamma, planning_step)


    def __call__(self, env: GridWorld,
                epsilon: float,
                alpha: float, 
                gamma: float,
                planning_step: int) -> object:
        return DynaQ(env, epsilon, alpha, gamma, planning_step)


    def _model_learning(self, state: np.ndarray, 
            action: int, next_state: np.ndarray,
            reward: float) -> None:
        '''
        Model learning

        Params
        ------
        state: state of the agent
        action: action taken at @state
        next_state: next state according to @state
        reward: reward taken at @next_state
        '''
        state_ = (state[0], state[1])
        if state_ not in self.model.keys():
            self.model[state_] = dict()
        self.model[state_][action] = next_state, reward


    def _search_control(self) -> Tuple[Tuple[int, int], int, np.ndarray, float]:
        '''
        Search control

        Return
        ------
        '''
        states = list(self.model.keys())
        state = states[np.random.choice(len(states))]
        actions = list(self.model[state].keys())
        action = actions[np.random.choice(len(actions))]
        next_state, reward = self.model[state][action]
        return state, action, next_state, reward


    def run(self) -> int:
        '''
        Perform an episode

        Return
        ------
        n_steps: number of steps of the episode
        '''
        state = self.reset()
        n_steps = 0

        while True:
            action = self._epsilon_greedy(state)
            next_state, reward, terminated = self.env.step(action)
            self._update_Q(state, action, next_state, reward)
            self._model_learning(state, action, next_state, reward)
            for _ in range(self.planning_step):
                state_, action_, next_state_, reward_ = self._search_control()
                self._update_Q(state_, action_, next_state_, reward_)
            
            state = next_state
            n_steps += 1

            if terminated:
                break

        return n_steps


class DynaQPlus(DynaQ):
    '''
    Dyna-Q+ agent
    '''

    def __init__(self, env: GridWorld,
                epsilon: float,
                alpha: float,
                gamma: float,
                planning_step: int,
                kappa: float) -> None:
        '''
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        planning_step: number of planning steps
        kappa: exploration bonus param
        '''
        super().__init__(env, epsilon, alpha, gamma, planning_step)
        self.kappa = kappa
        self.current_step = 0


    def __call__(self, env: GridWorld,
                epsilon: float,
                alpha: float, 
                gamma: float,
                planning_step: int,
                kappa: float) -> object:
        return DynaQPlus(env, epsilon, alpha, gamma, planning_step, kappa)


    def _model_learning(self, state: np.ndarray, 
            action: int, next_state: np.ndarray,
            reward: float) -> None:
        '''
        Model learning

        Params
        ------
        state: state of the agent
        action: action taken at @state
        next_state: next state according to @state
        reward: reward taken at @next_state
        '''
        self.current_step += 1
        state_ = (state[0], state[1])
        if state_ not in self.model.keys():
            self.model[state_] = dict()
            for action_ in self.env.action_space:
                if action_ != action:
                    # lead back to the same state with a reward of zero
                    self.model[state_][action_] = state, 0, 1
        self.model[state_][action] = next_state, reward, self.current_step


    def _search_control(self) -> Tuple[Tuple[int, int], int, np.ndarray, float]:
        '''
        Search control

        Return
        ------
        '''
        states = list(self.model.keys())
        state = states[np.random.choice(len(states))]
        actions = list(self.model[state].keys())
        action = actions[np.random.choice(len(actions))]
        next_state, reward, last_tried = self.model[state][action]
        reward += self.kappa * np.sqrt(self.current_step - last_tried)
        return state, action, next_state, reward


class PrioritizedSweeping(DynaQ):
    '''
    Prioritized sweeping agent
    '''

    def __init__(self, env: GridWorld,
                epsilon: float,
                alpha: float,
                gamma: float,
                planning_step: int,
                theta: float) -> None:
        '''
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        planning_step: number of planning steps
        theta: 
        '''
        super().__init__(env, epsilon, alpha, 
            gamma, planning_step)
        self.theta = theta
        self.pqueue = PriorityQueue()
        self.predecessor_pairs = dict()


    def __call__(self, env: GridWorld,
                epsilon: float,
                alpha: float, 
                gamma: float,
                planning_step: int,
                theta: float) -> object:
        return PrioritizedSweeping(env, epsilon, alpha, 
            gamma, planning_step, theta)


    def _model_learning(self, state: np.ndarray, 
            action: int, next_state: np.ndarray,
            reward: float) -> None:
        '''
        Model learning

        Params
        ------
        state: state of the agent
        action: action taken at @state
        next_state: next state according to @state
        reward: reward taken at @next_state
        '''
        super()._model_learning(state, action, next_state, reward)

        next_state_ = (next_state[0], next_state[1])
        if next_state_ not in self.predecessor_pairs.keys():
            self.predecessor_pairs[next_state_] = []
        self.predecessor_pairs[next_state_].append((state, action))


    def _search_control(self) -> Tuple[Tuple[int, int], int, np.ndarray, float]:
        state, action = self.pqueue.pop()
        next_state, reward = self.model[state][action]
        return state, action, next_state, reward


    # return state, action, reward predicted to lead to a state
    def _get_predecessor_pairs(self, state: np.ndarray):
        '''
        
        '''
        predecessors_ = []
        state_ = (state[0], state[1])
        if state_ in self.predecessor_pairs.keys():
            for pre_state, pre_action in self.predecessor_pairs[state_]:
                pre_state_ = (pre_state[0], pre_state[1])
                predecessors_.append([pre_state, pre_action, 
                    self.model[pre_state_][pre_action][1]])
        return predecessors_


    def _cal_priority(self, state: np.ndarray,
                    action: int,
                    next_state: np.ndarray, 
                    reward: float) -> float:
        '''
        Compute the priority 

        Params
        ------
        state: state of the agent
        action: action taken at @state
        next_state: next state according to @state
        reward: reward taken at @next_state
        '''
        return np.abs(reward + self.gamma * np.max(self.value_function[next_state[0], next_state[1], :]) 
            - self.value_function[state[0], state[1], action])


    def run(self) -> int:
        '''
        Perform an episode

        Return
        ------
        n_updates: number of updates of the episode
        '''
        state = self.reset()
        n_steps = 0
        n_updates = 0

        while True:
            action = self._epsilon_greedy(state)
            next_state, reward, terminated = self.env.step(action)
            self._model_learning(state, action, next_state, reward)

            priority = self._cal_priority(state, action, next_state, reward)
            if priority > self.theta:
                # since the heap in heapq is a min-heap
                state_ = (state[0], state[1])
                self.pqueue.push((state_, action), -priority)

            planning_step_count = 0
            while planning_step_count < self.planning_step and not self.pqueue.is_empty():
                state_, action_, next_state_, reward_ = self._search_control()
                self._update_Q(state_, action_, next_state_, reward_)

                for pre_state_, pre_action_, pre_reward_ in self._get_predecessor_pairs(state_):
                    priority_ = self._cal_priority(pre_state_, pre_action_, state_, pre_reward_)
                    if priority_ > self.theta:
                        pre_state__ = (pre_state_[0], pre_state_[1])
                        self.pqueue.push((pre_state__, pre_action_), -priority_)
                planning_step_count += 1

            state = next_state
            n_steps += 1
            n_updates += planning_step_count + 1

            if terminated:
                break

        return n_updates


def dyna_maze():
    height = 6
    width = 9
    start_state = (2, 0)
    terminal_states = [(0, 8)]
    obstacles = [(0, 7), (1, 7), (2, 7), (1, 2), (2, 2), (3, 2), (4, 5)]
    maze = GridWorld(height, width, start_state, 
        terminal_states, obstacles=obstacles)

    n_runs = 30
    n_episodes = 50 
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    planning_steps = [0, 5, 50]
    n_steps = np.zeros((len(planning_steps), n_episodes))

    for _ in trange(n_runs):
        for i, planning_step in enumerate(planning_steps):
            agent = DynaQ(maze, epsilon, alpha, gamma, planning_step)

            for ep in range(n_episodes):
                n_steps[i, ep] += agent.run()

    n_steps /= n_runs

    for i in range(len(planning_steps)):
        plt.plot(n_steps[i, :], label=f'{planning_steps[i]} planning steps')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend()

    plt.savefig('./dyna_maze.png')
    plt.close()


def blocking_maze():
    height = 6
    width = 9
    start_state = (5, 3)
    terminal_states = [(0, 8)]
    obstacles = [(3, i) for i in range(8)]
    new_obstacles = [(3, i) for i in range(1, 9)]

    n_runs = 20
    alpha = 1
    epsilon = 0.1
    gamma = 0.95
    planning_step = 10
    obstacles_change_step = 1000
    max_step = 3000
    kappa = 1e-4

    maze = GridWorld(height, width, start_state, terminal_states)
    methods = [
        {
            'name' :'Dyna-Q',
            'agent': DynaQ,
            'params': [maze, epsilon, alpha, gamma, planning_step]
        },
        {
            'name': 'Dyna-Q+',
            'agent': DynaQPlus,
            'params': [maze, epsilon, alpha, gamma, planning_step, kappa]
        }
    ]
    rewards = np.zeros((n_runs, len(methods), max_step))
    
    for i, method in enumerate(methods):
        print(method['name'])

        for run in trange(n_runs):
            maze.set_obstacles(obstacles)
            agent = method['agent'](*method['params'])

            step = 0
            last_step = 0
            while step < max_step:
                step += agent.run()
    
                rewards[run, i, last_step: step] = rewards[run, i, last_step]
                rewards[run, i, min(step, max_step - 1)] = rewards[run, i, last_step] + 1
                last_step = step

                if step > obstacles_change_step:
                    maze.set_obstacles(new_obstacles)
        time.sleep(0.1)

    rewards = np.mean(rewards, axis=0)

    for i, method in enumerate(methods):
        plt.plot(rewards[i, :], label=method['name'])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.savefig('./blocking_maze.png')
    plt.close()


def shortcut_maze():
    height = 6
    width = 9
    start_state = (5, 3)
    terminal_states = [(0, 8)]
    obstacles = [(3, i) for i in range(1, 9)]
    new_obstacles = [(3, i) for i in range(1, 8)]

    n_runs = 5
    alpha = 1
    epsilon = 0.1
    gamma = 0.95
    planning_step = 50
    obstacles_change_step = 3000
    max_step = 6000
    kappa = 1e-3

    maze = GridWorld(height, width, start_state, terminal_states)
    methods = [
        {
            'name' :'Dyna-Q',
            'agent': DynaQ,
            'params': [maze, epsilon, alpha, gamma, planning_step]
        },
        {
            'name': 'Dyna-Q+',
            'agent': DynaQPlus,
            'params': [maze, epsilon, alpha, gamma, planning_step, kappa]
        }
    ]
    rewards = np.zeros((n_runs, len(methods), max_step))

    for i, method in enumerate(methods):
        print(method['name'])

        for run in trange(n_runs):
            maze.set_obstacles(obstacles)
            agent = method['agent'](*method['params'])

            step = 0
            last_step = 0
            while step < max_step:
                step += agent.run()
    
                rewards[run, i, last_step: step] = rewards[run, i, last_step]
                rewards[run, i, min(step, max_step - 1)] = \
                    rewards[run, i, last_step] + 1
                last_step = step

                if step > obstacles_change_step:
                    maze.set_obstacles(new_obstacles)
        time.sleep(0.1)

    rewards = np.mean(rewards, axis=0)

    for i, method in enumerate(methods):
        plt.plot(rewards[i, :], label=method['name'])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.savefig('./shortcut_maze.png')
    plt.close()


def is_optimal_solution(agent: DynaAgent, resolution: int) -> bool:
    max_steps = 14 * resolution * 1.2
    state = agent.reset()
    n_steps = 0
    while True:
        action = np.argmax(agent.value_function[state[0], state[1], :])
        state, _, terminated = agent.env.step(action)
        n_steps += 1

        if terminated:
            break
        if n_steps > max_steps:
            return False
    return True


def mazes():
    height = 6
    width = 9
    start_state = (2, 0)
    terminal_states = [(0, 8)]
    obstacles = [(0, 7), (1, 7), (2, 7), (1, 2), (2, 2), (3, 2), (4, 5)]
    maze = GridWorld(height, width, start_state, 
        terminal_states, obstacles=obstacles)
    resolutions = range(1, 6)

    n_runs = 5
    alpha = 0.5
    epsilon = 0.1
    gamma = 0.95
    planning_step = 5
    theta = 0.0001
    methods = [
        {
            'name': 'Dyna-Q',
            'agent': DynaQ,
            'params': [epsilon, alpha, gamma, planning_step]
        },
        {
            'name': 'Prioritized Sweeping',
            'agent': PrioritizedSweeping,
            'params': [epsilon, alpha, gamma, planning_step, theta]
        }
    ]
    updates = np.zeros((n_runs, len(methods), len(resolutions)))

    for i, method in enumerate(methods):
        print(method['name'])

        for run in range(n_runs):

            for res in resolutions:
                maze_ = maze.extend(res)
                agent = method['agent'](maze_, *method['params'])

                print(f'run = {run}, maze size = {maze_.height * maze_.width}')
                steps = []

                while True:
                    steps.append(agent.run())
                    if is_optimal_solution(agent, res):
                        break

                updates[run, i, res - 1] = np.sum(steps)
        time.sleep(0.1)

    updates = np.mean(updates, axis=0)
    updates[0, :] *= planning_step + 1

    for i, method in enumerate(methods):
        plt.plot(np.arange(1, len(resolutions) + 1), updates[i, :], label=method['name'])
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
