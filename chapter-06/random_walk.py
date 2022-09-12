import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
import string
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import RandomWalk


def get_true_value(env: RandomWalk) -> np.ndarray:
    '''
    Compute true values
    '''
    true_value = np.array([1.0 * x / (env.n_states + 1) 
        for x in range(1, env.n_states + 1)])
    return true_value


class Agent(ABC):
    '''
    Agent abstract class
    '''

    def __init__(self, env: RandomWalk,
                value_function: np.ndarray,
                alpha: float, gamma: float,
                batch_update: bool=False) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        value_function: value function
        alpha: step size param
        gamma: discount factor
        batch_update: whether using batch updating
        '''
        self.env = env
        self.value_function = value_function
        self.alpha = alpha
        self.gamma = gamma
        self.batch_update = batch_update


    @abstractmethod
    def __call__(self, env: RandomWalk, 
                value_function: np.ndarray,
                alpha: float, gamma: float,
                batch_update: bool=False) -> object:
        pass


    def reset(self) -> None:
        '''
        Reset agent
        '''
        self.env.reset()


    def random_policy(self) -> int:
        '''
        Policy choosing actions randomly

        Return
        ------
        action: chosen action
        '''
        action = np.random.choice(self.env.action_space)
        return action


    @abstractmethod
    def run(self) -> Tuple[List[int], List[float]]:
        '''
        Perform an episode
        '''
        pass


class TemporalDifference(Agent):
    '''
    Temporal Difference agent
    '''

    def __init__(self, env: RandomWalk,
                value_function: np.ndarray,
                alpha: float, gamma: float,
                batch_update: bool=False) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        value_function: value function
        alpha: step size param
        gamma: discount factor
        batch_update: whether using batch updating
        '''
        super().__init__(env, value_function,
            alpha, gamma, batch_update)


    def __call__(self, env: RandomWalk, 
                value_function: np.ndarray,
                alpha: float, gamma: float,
                batch_update: bool=False) -> object:
        return TemporalDifference(env, value_function,
                    alpha, gamma, batch_update)


    def run(self) -> Tuple[List[int], List[float]]:
        '''
        Perform an episode

        Return
        ------
        states: state history
        rewards: reward history
        '''
        self.reset()
        states = [self.env.state]
        rewards = [0]

        while True:
            action = self.random_policy()
            state = self.env.state
            next_state, reward, terminated = self.env.step(action)
            states.append(next_state)
            rewards.append(reward)

            if not self.batch_update:
                self.value_function[state] += self.alpha * (reward + self.gamma * \
                    self.value_function[next_state] - self.value_function[state])

            if terminated:
                break

        return states, rewards


class MonteCarlo(Agent):
    '''
    Monte Carlo agent
    '''

    def __init__(self, env: RandomWalk,
                value_function: np.ndarray,
                alpha: float, gamma: float,
                batch_update: bool=False) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        value_function: value function
        alpha: step size param
        gamma: discount factor
        batch_update: whether using batch updating
        '''
        super().__init__(env, value_function,
            alpha, gamma, batch_update)


    def __call__(self, env: RandomWalk, 
                value_function: np.ndarray,
                alpha: float, gamma: float,
                batch_update: bool=False) -> object:
        return MonteCarlo(env, value_function,
                    alpha, gamma, batch_update)


    def run(self) -> Tuple[List[int], List[float]]:
        '''
        Perform an episode

        Return
        ------
        states: state history
        rewards: reward history
        '''
        self.reset()
        states = [self.env.state]
        rewards = [0]

        while True:
            action = self.random_policy()
            next_state, reward, terminated = self.env.step(action)
            for t in range(len(rewards)):
                rewards[t] += np.power(self.gamma, len(rewards) - t) * reward
            states.append(next_state)
            rewards.append(reward)

            if terminated:
                break

        # the return at each state is equal to the reward at the terminal state.
        if not self.batch_update:
            for state, reward in zip(states[:-1], rewards[:-1]):
                self.value_function[state] += self.alpha * \
                    (reward - self.value_function[state])

        return states, rewards


def plot_state_values(env: RandomWalk,
                    true_value: np.ndarray,
                    n_eps: int, alpha: float,
                    gamma: float) -> None:
    '''
    Plot state values using TD

    Params
    ------
    env: RandomWalk env
    true_value: true values
    n_eps: number of episodes
    alpha: step size param
    gamma: discount factor
    '''
    value_function = np.full(env.n_states + 2, 0.5)
    value_function[0] = value_function[-1] = 0
    eps_plot = [0, 1, 10, 100]

    state_labels = list(string.ascii_uppercase)[:env.n_states]
    plt.plot(state_labels, true_value, label='true values')

    temporal_difference = TemporalDifference(env, value_function, alpha, gamma)
    for ep in range(n_eps + 1):
        if ep in eps_plot:
            plt.plot(state_labels, value_function[1:-1], label=str(ep) + ' episodes')
        temporal_difference.run()

    plt.xlabel('State')
    plt.ylabel('Estimated value')
    plt.legend()


def plot_rmse(env: RandomWalk, true_value: np.ndarray,
            n_eps: int, gamma: float) -> None:
    '''
    Plot RMSE

    Params
    ------
    env: RandomWalk env
    true_value: true values
    n_eps: number of episodes
    gamma: discount factor
    '''
    value_function = np.full(env.n_states + 2, 0.5)
    value_function[0] = value_function[-1] = 0
    methods = [
        {
            'name': 'TD',
            'alphas': [0.05, 0.1, 0.15],
            'agent': TemporalDifference,
            'linestyle': 'solid'
        },
        {
            'name': 'MC',
            'alphas': [0.01, 0.02, 0.03, 0.04],
            'agent': MonteCarlo,
            'linestyle': 'dashdot'
        }
    ]
    n_runs = 100

    for method in methods:
        for alpha in method['alphas']:
            print(f'{method["name"]} method, alpha={alpha}', end='')

            total_errors = np.zeros(n_eps)
            for _ in trange(n_runs):
                value_function_ = value_function.copy()
                agent = method['agent'](env, value_function_, alpha, gamma)
                errors = []

                for _ in range(n_eps):
                    rmse = np.sqrt(np.sum(np.power(value_function_[1:-1] - 
                        true_value, 2) / env.n_states))
                    errors.append(rmse)
                    agent.run()

                total_errors += np.asarray(errors)
            total_errors /= n_runs
            plt.plot(total_errors, label=method['name'] + ', alpha = %.02f' 
                % (alpha), linestyle=method['linestyle'])
            print()
    plt.xlabel('Episodes')
    plt.ylabel('RMS')
    plt.legend()


def plot_rmse_batch_updating(env: RandomWalk,
            true_value: np.ndarray, n_eps: int,
            alpha: float, gamma: float) -> None:
    value_function = np.full(env.n_states + 2, -1.0)
    value_function[0] = 0
    value_function[-1] = 1
    methods = [
        {
            'name': 'TD',
            'agent': TemporalDifference,
        },
        {
            'name': 'MC',
            'agent': MonteCarlo,
        }
    ]
    n_runs = 100

    for method in methods:
        print(f'{method["name"]} method', end='')
        total_errors = np.zeros(n_eps)
        for _ in trange(n_runs):
            value_function_ = value_function.copy()
            errors = []
            state_history = []
            reward_history = []
            agent = method['agent'](env, value_function_, alpha, gamma, True)

            for _ in range(n_eps):
                states, rewards = agent.run()
                state_history.append(states)
                reward_history.append(rewards)

                while True:
                    error = np.zeros(env.n_states + 2)

                    for states_, rewards_ in zip(state_history, reward_history):
                        for t in range(len(states_) - 1):
                            state = states_[t]
                            next_state = states_[t + 1]
                            if method['name'] == 'TD':
                                reward = rewards_[t]
                                error[state] += alpha * (reward + gamma * 
                                    value_function_[next_state] - value_function_[state])
                            else:
                                return_ = rewards_[t]
                                error[state] += alpha * (return_ - value_function_[state])
                    if np.sum(np.abs(error)) < 1e-3:
                        break
                    value_function_ += error

                rmse = np.sqrt(np.sum(np.power(value_function_[1:-1] 
                    - true_value, 2)) / env.n_states)
                errors.append(rmse)
            total_errors += np.asarray(errors)
        total_errors /= n_runs
        plt.plot(total_errors, label=method['name'])
        print()
    plt.xlabel('Episodes')
    plt.ylabel('RMS')
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.legend()
    plt.savefig('./random_walk_batch_updating.png')
    plt.close()


if __name__ == '__main__':
    n_states = 5
    start_state = 3
    terminal_states = [0, n_states + 1]
    reward_space = [1, 0]
    env = RandomWalk(n_states, start_state, terminal_states, reward_space=reward_space)
    n_eps = 100
    alpha = 0.1
    gamma = 1
    true_value = get_true_value(env)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plot_state_values(env, true_value, n_eps, alpha, gamma)

    plt.subplot(1, 2, 2)
    plot_rmse(env, true_value, n_eps, gamma)
    plt.tight_layout()
    plt.savefig('./random_walk.png')
    plt.close()

    print('Batch updating')
    batch_alpha = 0.001
    plot_rmse_batch_updating(env, true_value, n_eps, batch_alpha, gamma)
