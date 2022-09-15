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
                gamma: float, expected: bool) -> None:
        '''
        Params
        ------
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        expected: whether using expected update (for Expected Sarsa)
        '''
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.expected = expected
        self.value_function = np.zeros((env.height, 
            env.width, len(env.action_space)))


    @abstractmethod
    def __call__(self, env: GridWorld,
                epsilon: float, alpha: float, 
                gamma: float, expected: bool) -> object:
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


    def _update_Q(self, state: np.ndarray, 
                action: int, target: float) -> None:
        '''
        Update state-action value function

        Params
        ------
        state: state of the agent
        action: action taken at state @state
        target: target of the update
        '''
        estimate = self.value_function[state[0], state[1], action]
        self.value_function[state[0], state[1], action] \
            += self.alpha * (target - estimate)


    @abstractmethod
    def run(self) -> float:
        pass


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
            gamma: float, expected: bool=None) -> None:
        '''
        Params
        ------
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        expected: whether using expected update (for Expected Sarsa)
        '''
        super().__init__(env, epsilon, alpha, gamma, expected)


    def __call__(self, env: GridWorld,
                epsilon: float, alpha: float, 
                gamma: float, expected: bool=None) -> object:
        return QLearning(env, epsilon, alpha, gamma, expected)


    def run(self) -> float:
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
            target = reward + self.gamma * np.max(self.value_function[next_state[0], next_state[1], :])
            self._update_Q(state, action, target)
            state = next_state

            if terminated:
                break

        return total_reward


class Sarsa(Agent):
    '''
    Sarsa - Expected Sarsa agent
    '''

    def __init__(self, env: GridWorld, 
                epsilon: float, alpha: float, 
                gamma: float, expected: bool=None) -> None:
        '''
        Params
        ------
        env: GridWorld env
        epsilon: exploration param
        alpha: step size param
        gamma: discount factor
        expected: whether using expected update (for Expected Sarsa)
        '''
        super().__init__(env, epsilon, alpha, gamma, expected)


    def __call__(self, env: GridWorld,
                epsilon: float, alpha: float, 
                gamma: float, expected: bool=False) -> object:
        return Sarsa(env, epsilon, alpha, gamma, expected)


    def run(self) -> float:
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
            if self.expected:
                next_state_exp_value = self.epsilon / len(self.env.action_space) \
                    * np.sum(self.value_function[next_state[0], next_state[1], :])
                next_state_exp_value += (1 - self.epsilon) \
                    * np.max(self.value_function[next_state[0], next_state[1], :])
                target = reward + self.gamma * next_state_exp_value 
            else:
                target = reward + self.gamma * self.value_function[next_state[0], next_state[1], next_action]
            self._update_Q(state, action, target)
            state = next_state
            action = next_action

            if terminated:
                break

        return total_reward


def q_learning_sarsa(env: GridWorld, 
        epsilon: float, gamma: float) -> None:
    '''
    Plot comparison of Q-learning - Sarsa

    Params
    ------
    env: GridWorld env
    epsilon: exploration param
    gamma: discount factor
    '''
    n_runs = 50
    n_eps = 500
    alpha = 0.5

    methods = [
        {
            'name': 'Q-learning',
            'agent': QLearning,
        },
        {
            'name': 'Sarsa',
            'agent': Sarsa
        }
    ]

    rewards = np.zeros((len(methods), n_eps))

    for method_idx, method in enumerate(methods):
        print(method['name'])
        for _ in trange(n_runs):
            agent = method['agent'](env, epsilon, alpha, gamma)

            for ep in range(n_eps):
                rewards[method_idx, ep] += agent.run()

    rewards /= n_runs

    for i, method in enumerate(methods):
        plt.plot(rewards[i], label=method['name'])

    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('./cliff-walking-q-learning-sarsa.png')
    plt.close()


def q_learning_sarsa_expected_sarsa(env: GridWorld, 
        epsilon: float, gamma: float) -> None:
    '''
    Plot comparison of Q-learning - Sarsa - Expected Sarsa

    Params
    ------
    env: GridWorld env
    epsilon: exploration param
    gamma: discount factor
    '''
    alphas = np.arange(0.1, 1.1, 0.1)
    n_runs = 10
    n_eps = 1000

    methods = [
        {
            'name': 'Q-learning',
            'agent': QLearning,
            'expected': False
        },
        {
            'name': 'Sarsa',
            'agent': Sarsa,
            'expected': False
        },
        {
            'name': 'Expected Sarsa',
            'agent': Sarsa,
            'expected': True
        }
    ]

    performace_types = ['Asymptotic', 'Interim']
    performace_styles = ['solid', 'dashed']
    performance = np.zeros((len(methods), len(performace_types), len(alphas)))

    for method_idx, method in enumerate(methods):
        for alpha_idx, alpha in enumerate(alphas):
            name = method['name']
            expected = method['expected']
            print(f'{name}, alpha={alpha}')

            for _ in trange(n_runs):
                agent = method['agent'](env, epsilon, alpha, gamma, expected)

                for ep in range(n_eps):
                    rewards = agent.run()
                    performance[method_idx, 0, alpha_idx] += rewards

                    if ep < 100:
                        performance[method_idx, 1, alpha_idx] += rewards

    performance[:, 0, :] /= n_eps * n_runs
    performance[:, 1, :] /= 100 * n_runs

    for pfm_idx, pfm_type in enumerate(performace_types):
        for method_idx, method in enumerate(methods):
            label = pfm_type + ' ' + method['name']
            plt.plot(alphas, performance[method_idx, pfm_idx, :], label=label, 
                linestyle=performace_styles[pfm_idx])
    plt.xlabel('alpha')
    plt.ylabel('Reward per episode')
    plt.legend()

    plt.savefig('./cliff-walking-q-learning-sarsa-expected-sarsa.png')
    plt.close()


if __name__ == '__main__':
    height = 4
    width = 13
    start_state = (3, 0)
    terminal_states = [(3, 12)]
    cliff = [(3, x) for x in range(1, 12)]
    epsilon = 0.1
    gamma = 1
    env = GridWorld(height, width, start_state, terminal_states, cliff=cliff)
    q_learning_sarsa(env, epsilon, gamma)
    q_learning_sarsa_expected_sarsa(env, epsilon, gamma)
