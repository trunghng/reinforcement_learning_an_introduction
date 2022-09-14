import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import GridWorld


class Agent(ABC):
    '''
    Agent abstract class
    '''

    def __init__(self, env: GridWorld, 
                epsilon: float, alpha: float,
                gamma: float, n_eps: int) -> None:
        '''
        Params
        ------
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        n_eps: number of episodes
        '''
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_eps = n_eps


    def _reset(self) -> np.ndarray:
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
        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.env.action_space)
        else:
            state = self.env.state
            max_value = self.value_function[state[0], state[1], :].max()
            action = np.random.choice(np.flatnonzero(
                self.value_function[state[0], state[1], :] == max_value))
        return action


    @abstractmethod
    def _run_episode(self) -> float:
        pass


    def run(self) -> List[int]:
        n_steps = []
        self.value_function = np.zeros((self.env.height, self.env.width, 
            len(self.env.action_space)))

        for ep in trange(self.n_eps):
            n_steps.append(self._run_episode())

        return n_steps


    def print_optimal_policy(self) -> None:
        for x in range(self.env.height):
            optimal_policy_row = []
            for y in range(self.env.width):
                if self.env.terminated(np.array([x, y])):
                    optimal_policy_row.append('G')
                    continue
                best_action = np.argmax(self.value_function[x, y, :])
                if best_action == 0:
                    optimal_policy_row.append('U')
                elif best_action == 1:
                    optimal_policy_row.append('R')
                elif best_action == 2:
                    optimal_policy_row.append('D')
                elif best_action == 3:
                    optimal_policy_row.append('L')
            print(optimal_policy_row)


class Sarsa(Agent):
    '''
    Sarsa agent
    '''

    def __init__(self, env: GridWorld, 
                epsilon: float, alpha: float, 
                gamma: float, n_eps: int) -> None:
        super().__init__(env, epsilon, alpha, gamma, n_eps)


    def _run_episode(self) -> int:
        '''
        Perform an episode 

        Return
        ------
        n_steps: number of steps of the episode
        '''
        state = self._reset()
        action = self._epsilon_greedy(state)
        n_steps = 0

        while True:
            n_steps += 1
            next_state, reward, terminated = self.env.step(action)
            next_action = self._epsilon_greedy(next_state)
            self.value_function[state[0], state[1], action] += \
                self.alpha * (reward + self.gamma * self.value_function[next_state[0], \
                next_state[1], next_action] - self.value_function[state[0], state[1], action])
            state = next_state
            action = next_action

            if terminated:
                break

        return n_steps


if __name__ == '__main__':
    height = 7
    width = 10
    wind_dist = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    start_state = (3, 0)
    terminal_states = [(3, 7)]
    env = GridWorld(height, width, start_state, terminal_states, wind_dist=wind_dist)
    n_eps = 600
    epsilon = 0.1
    alpha = 0.5
    gamma = 1

    sarsa = Sarsa(env, epsilon, alpha, gamma, n_eps)

    n_steps = sarsa.run()
    n_steps = np.add.accumulate(time_steps)

    plt.plot(time_steps, np.arange(1, len(time_steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./windy_gridworld.png')
    plt.close()

    sarsa.print_optimal_policy()
