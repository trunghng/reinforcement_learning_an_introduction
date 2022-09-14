import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from abc import ABC, abstractmethod

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
                gamma: float, n_runs: int,
                n_eps: int) -> None:
        '''
        Params
        ------
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        n_runs: number of runs
        n_eps: number of episodes
        '''
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_runs = n_runs
        self.n_eps = n_eps


    @abstractmethod
    def __call__(self, env: GridWorld,
                epsilon: float, alpha: float, 
                gamma: float, n_runs: int,
                n_eps: int) -> object:
        pass


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


    def run(self) -> np.ndarray:
        rewards = np.zeros(self.n_eps)

        for _ in trange(self.n_runs):
            self.value_function = np.zeros((self.env.height, self.env.width, 
                len(self.env.action_space)))

            for ep in range(self.n_eps):
                rewards[ep] += self._run_episode()

        rewards /= self.n_runs
        return rewards


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


class QLearning(Agent):
    '''
    Q-learning agent
    '''

    def __init__(self, env: GridWorld, 
            epsilon: float, alpha: float,
            gamma: float, n_runs: int,
            n_eps: int) -> None:
        super().__init__(env, epsilon, alpha, gamma, n_runs, n_eps)


    def __call__(self, env: GridWorld,
                epsilon: float, alpha: float, 
                gamma: float, n_runs: int,
                n_eps: int) -> object:
        return QLearning(env, epsilon, alpha, gamma, n_runs, n_eps)


    def _run_episode(self) -> float:
        '''
        Perform an episode 

        Return
        ------
        total_reward: total reward of the episode
        '''
        state = self._reset()
        total_reward = 0

        while True:
            action = self._epsilon_greedy(state)
            next_state, reward, terminated = self.env.step(action)
            total_reward += reward
            self.value_function[state[0], state[1], action] += self.alpha \
                * (reward + self.gamma * np.max(self.value_function[next_state[0],\
                next_state[1], :]) - self.value_function[state[0], state[1], action])
            state = next_state

            if terminated:
                break

        return total_reward


class Sarsa(Agent):
    '''
    Sarsa agent
    '''

    def __init__(self, env: GridWorld, 
                epsilon: float, alpha: float, 
                gamma: float, n_runs: int,
                n_eps: int) -> None:
        super().__init__(env, epsilon, alpha, gamma, n_runs, n_eps)


    def __call__(self, env: GridWorld,
                epsilon: float, alpha: float, 
                gamma: float, n_runs: int,
                n_eps: int) -> object:
        return Sarsa(env, epsilon, alpha, gamma, n_runs, n_eps)


    def _run_episode(self) -> float:
        '''
        Perform an episode 

        Return
        ------
        total_reward: total reward of the episode
        '''
        state = self._reset()
        action = self._epsilon_greedy(state)
        total_reward = 0

        while True:
            next_state, reward, terminated = self.env.step(action)
            total_reward += reward
            next_action = self._epsilon_greedy(next_state)
            self.value_function[state[0], state[1], action] += \
                self.alpha * (reward + self.gamma * self.value_function[next_state[0], \
                next_state[1], next_action] - self.value_function[state[0], state[1], action])
            state = next_state
            action = next_action

            if terminated:
                break

        return total_reward


if __name__ == '__main__':
    height = 4
    width = 13
    start_state = (3, 0)
    terminal_states = [(3, 12)]
    cliff = [(3, x) for x in range(1, 12)]
    env = GridWorld(height, width, start_state, terminal_states, cliff=cliff)
    n_runs = 50
    n_eps = 500
    epsilon = 0.1
    alpha = 0.5
    gamma = 1

    methods = [
        {
            'name': 'Q-learning',
            'agent': QLearning
        },
        {
            'name': 'Sarsa',
            'agent': Sarsa
        }
    ]

    rewards = []

    for method in methods:
        name = method['name']
        print(name)
        agent = method['agent'](env, epsilon, alpha, gamma, n_runs, n_eps)
        rewards_ = agent.run()
        rewards.append(rewards_)
        print(f'{name}\'s optimal policy:')
        agent.print_optimal_policy()

    for i, method in enumerate(methods):
        plt.plot(rewards[i], label=method['name'])

    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('./cliff_walking.png')
    plt.close()
