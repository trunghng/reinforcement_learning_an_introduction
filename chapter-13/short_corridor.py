import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import ShortCorridor


class REINFORCE:
    '''
    REINFORCE agent
    '''

    def __init__(self, env: ShortCorridor,
                gamma: float,
                epsilon: float,
                alpha: float, 
                n_eps: int) -> None:
        '''
        Parameters
        ------
        env: short-corridor env
        gamma: discount factor
        epsilon: exploration parameter
        alpha: step size
        n_eps: number of episodes
        '''
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_eps = n_eps
        self.theta = np.array([-1.47, 1.47])
        # feature mapping
        self.x = lambda s, a: np.array([int(a == self.env.action_space[1]), 
                                        int(a == self.env.action_space[0])])

    def __call__(self, env: ShortCorridor,
                gamma: float,
                epsilon: float,
                alpha: float, 
                n_eps: int):
        return REINFORCE(env, gamma, epsilon, alpha, n_eps)


    def preference(self, state: int, action: int) -> float:
        '''
        Linear action preferences
            h(s,a,theta) = theta.dot(x)
        '''
        feature_vector = self.x(state, action)
        return self.theta.dot(feature_vector)


    def pi(self, state: int, action: int) -> float:
        '''
        Softmax in action preferences
            pi(s,a,theta) = exp(h(s,a,theta)) / sum_{b}exp(h(s,b,theta))
        '''
        def _softmax(zi, z):
            zmax = np.max(z)
            return np.exp(zi - zmax) / np.sum(np.exp(z - zmax))
        
        action_pref = self.preference(state, action)
        action_prefs = []
        for action_ in self.env.action_space:
            action_prefs.append(self.preference(state, action_))

        return _softmax(action_pref, np.array(action_prefs))


    def select_action(self, state: int, pi: List[List[float]]) -> int:
        amin = np.argmin(pi[state])
        if pi[state][amin] < self.epsilon:
            pi[state][amin] = self.epsilon
            pi[state][1 - amin] = 1 - self.epsilon

        action = self.env.action_space[1] \
            if np.random.binomial(1, pi[state][self.env.action_space[1]]) \
            else self.env.action_space[0]
        return action


    def eligibility_vector(self, state: int, action: int) -> np.ndarray:
        '''
        Compute eligibility vector
            = x(s,a) - sum_{b}(pi(a|s) * x(s,b))
        '''
        mean_feature = 0
        for action_ in self.env.action_space:
            mean_feature += self.pi(state, action_) * self.x(state, action_)

        return self.x(state, action) - mean_feature


    def update(self, state: int,
                action: int, 
                return_: float, 
                step: int) -> None:
        '''
        Update theta
            theta := theta + alpha * gamma^t * G * eligibility_vector

        Parameters
        ------
        state: state of the agent at @step
        action: action taken at @step
        return_: discounted cumulative reward at @step
        step: time step
        '''
        eligibility_vector = self.eligibility_vector(state, action)
        a = self.alpha * (self.gamma**step) * return_ * eligibility_vector
        self.theta += self.alpha * (self.gamma**step) * return_ * eligibility_vector


    def run(self) -> np.ndarray:
        '''
        Perform a run

        Returns
        -------
        rewards: total reward on episodes
        '''
        rewards = np.zeros(self.n_eps)

        for ep in range(self.n_eps):
            state = self.env.reset()
            trajectory = []
            pi = [[self.pi(state_, action_) for action_ in self.env.action_space] \
                        for state_ in self.env.state_space]

            # Generate episode + compute cumulative rewards
            while True:
                action = self.select_action(state, pi)
                next_state, reward, terminated = self.env.step(action)
                for t in range(len(trajectory)):
                    trajectory[t][3] += self.gamma**(len(trajectory) - t) * reward
                trajectory.append([state, action, reward, reward])
                if terminated:
                    break
                state = next_state

            for t in range(len(trajectory)):
                state, action, reward, return_ = trajectory[t]
                self.update(state, action, return_, t)
                rewards[ep] += reward

        return rewards


class REINFORCEBaseline(REINFORCE):
    '''
    REINFORCE with baseline

        Here we choose the approximate state-value function 
            hat(v)(s,w) as the baseline and let hat(v) = w
    '''

    def __init__(self, env: ShortCorridor,
                gamma: float,
                epsilon: float,
                alpha: float,
                alpha_w: float,
                n_eps: int) -> None:
        '''
        Parameters
        ------
        env: short-corridor env
        gamma: discount factor
        epsilon: exploration parameter
        alpha: step size for theta update
        alpha_w: step size for w update
        n_eps: number of episodes
        '''
        super().__init__(env, gamma, epsilon, alpha, n_eps)
        self.alpha_w = alpha_w
        self.w = 0


    def __call__(self, env: ShortCorridor,
                gamma: float,
                epsilon: float,
                alpha_theta: float,
                alpha_w: float,
                n_ep: int):
        return REINFORCEBaseline(env, gamma, epsilon, 
            alpha_theta, alpha_w, n_eps)


    def update(self, state: int,
                action: int,
                return_: float,
                step: int) -> None:
        '''
        Update policy parameter theta, state-value weight w
            delta = return - w
            w := w + alpha_w * delta
            theta := theta + alpha_theta * gamma^t * delta * eligibility_vector

        Parameters
        ------
        state: state of the agent at @step
        action: action taken at @step
        return_: discounted cumulative reward at @step
        step: time step
        '''
        delta = return_ - self.w
        eligibility_vector = self.eligibility_vector(state, action)
        self.w += self.alpha_w * delta
        self.theta += self.alpha * self.gamma**step * delta * eligibility_vector


def reinforce_plot(env: ShortCorridor,
                gamma: float,
                epsilon: float,
                n_runs: int,
                n_eps: int) -> None:
    '''
    Plot REINFORCE

    Parameters
    ----------
    env: short-corridor env
    gamma: discount factor
    epsilon: exploration parameter
    n_runs: number of runs
    n_eps: number of episodes
    '''
    alphas = [2e-3, 2e-4, 2e-5]
    rewards = np.zeros((len(alphas), n_runs, n_eps))
    colors = ['blue', 'red', 'green']
    labels = ['2e-3', '2e-4', '2e-5']

    for alpha_idx, alpha in enumerate(alphas):
        print(f'REINFORCE, alpha={labels[alpha_idx]}')

        for run in trange(n_runs):
            reinforce = REINFORCE(env, gamma, epsilon, alpha, n_eps)
            rewards[alpha_idx, run, :] = reinforce.run()

    for i in range(len(alphas)):
        plt.plot(np.arange(n_eps), rewards[i].mean(axis=0), \
            color=colors[i], label=r'$\alpha=$' + labels[i])

    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.savefig('./short-corridor-reinforce.png')
    plt.close()


def reinforce_baseline_plot(env: ShortCorridor,
                gamma: float,
                epsilon: float,
                n_runs: int,
                n_eps: int) -> None:
    '''
    Plot REINFORCE w/ baseline

    Parameters
    ----------
    env: short-corridor env
    gamma: discount factor
    epsilon: exploration parameter
    n_runs: number of runs
    n_eps: number of episodes
    '''
    alpha = 2e-4
    alpha_theta = (2**4) * alpha
    alpha_w = (2**7) * alpha

    methods = [
        {
            'agent': REINFORCEBaseline,
            'name': 'REINFORCE with baseline',
            'params': [env, gamma, epsilon, alpha_theta, alpha_w, n_eps]
        },
        {
            'agent': REINFORCE,
            'name': 'REINFORCE',
            'params': [env, gamma, epsilon, alpha, n_eps]
        }
    ]

    rewards = np.zeros((len(methods), n_runs, n_eps))

    for i, method in enumerate(methods):
        print(f'{method["name"]}')

        for run in trange(n_runs):
            agent = method['agent'](*method['params'])
            rewards[i, run, :] = agent.run()

    for i, method in enumerate(methods):
        plt.plot(np.arange(n_eps), rewards[i].mean(axis=0), label=method['name'])

    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.savefig('./short-corridor-reinforce-baseline.png')
    plt.close()


if __name__ == '__main__':
    n_states = 3
    start_state = 0 
    terminal_state = 3
    switched_states = [1]
    env = ShortCorridor(n_states, start_state, terminal_state, switched_states)
    gamma = 1
    epsilon = 0.05
    n_runs = 100
    n_eps = 1000

    reinforce_plot(env, gamma, epsilon, n_runs, n_eps)
    reinforce_baseline_plot(env, gamma, epsilon, n_runs, n_eps)
