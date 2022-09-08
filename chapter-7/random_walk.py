import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import RandomWalk


def get_true_value(env: RandomWalk, gamma: float) -> np.ndarray:
    '''
    Calculate true value of @env by Bellman equations

    Params
    ------
    env: RandomWalk env
    gamma: discount factor

    Return
    ------
    true_value: true value of all of the states
    '''
    P = np.zeros((env.n_states, env.n_states))
    r = np.zeros((env.n_states + 2, ))
    true_value = np.zeros((env.n_states + 2, ))
    env.reset()
    
    for state in env.state_space:
        trajectory = []

        for action in env.action_space:
            next_state, reward, terminated = env.step(action, state)
            trajectory.append((action, next_state, reward, terminated))

        for action, next_state, reward, terminated in trajectory:
            if not terminated:
                P[state - 1, next_state - 1] = env.transition_probs[action] * 1
                r[next_state] = reward
        
    u = np.zeros((env.n_states, ))
    u[0] = env.transition_probs[-1] * 1 * (-1 + gamma * env.reward_space[0])
    u[-1] = env.transition_probs[1] * 1 * (1 + gamma * env.reward_space[2])

    r = r[1:-1]
    true_value[1:-1] = np.linalg.inv(np.identity(env.n_states) 
        - gamma * P).dot(0.5 * (P.dot(r) + u))
    true_value[0] = true_value[-1] = 0

    return true_value


class NStepTemporalDifference:
    '''
    n-step TD agent
    '''

    def __init__(self, env: RandomWalk,
                n: int, alpha: float,
                gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        n: number of steps
        alpha: step size param
        gamma: discount factor
        '''
        self.env = env
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.value_function = np.zeros(env.n_states + 2)


    def reset(self) -> None:
        '''
        Reset agent
        '''
        self.env.reset()


    def random_policy(self) -> int:
        '''
        Policy choosing action randomly

        Return
        ------
        action: chosen action
        '''
        action = np.random.choice(self.env.action_space)
        return action


    def run(self) -> None:
        '''
        Perform an episode
        '''
        self.reset()
        states = [self.env.state]
        rewards = [0] # dummy reward to save the next reward as R_{t+1}
        terminates = [False] # flag list to indicate whether S_t is a terminal state

        T = float('inf')
        t = 0

        while True:
            if t < T:
                action = self.random_policy()
                next_state, reward, terminated = self.env.step(action)
                states.append(next_state)
                rewards.append(reward)
                terminates.append(terminated)
                if terminated:
                    T = t + 1

            tau = t - self.n + 1 # updated state's time

            if tau >= 0:
                G = 0 # return
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += np.power(self.gamma, i - tau - 1) * rewards[i]
                if tau + self.n < T:
                    G += np.power(self.gamma, self.n) * \
                        self.value_function[states[tau + self.n]]
                if not terminates[tau]:
                    self.value_function[states[tau]] += self.alpha \
                        * (G - self.value_function[states[tau]])
            t += 1
            if tau == T - 1:
                break


if __name__ == '__main__':
    n_states = 19
    start_state = 10
    terminal_states = [0, n_states + 1]
    alphas = np.arange(0, 1.1, 0.1)
    gamma = 1
    env = RandomWalk(n_states, start_state, terminal_states)
    true_value = get_true_value(env, gamma)
    n_eps = 10
    n_runs = 100
    ns = np.power(2, np.arange(0, 10))

    errors = np.zeros((len(ns), len(alphas)))
    for n_i, n in enumerate(ns):
        for alpha_i, alpha in enumerate(alphas):
            print(f'n={n}, alpha={alpha}')

            for _ in trange(n_runs):
                n_step_td = NStepTemporalDifference(env, n, alpha, gamma)

                for _ in range(n_eps):
                    n_step_td.run()
                    values = np.array(n_step_td.value_function)
                    rmse = np.sqrt(np.sum(np.power(values - true_value, 2) / env.n_states))
                    errors[n_i, alpha_i] += rmse

    errors /= n_eps * n_runs

    for i in range(0, len(ns)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (ns[i]))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()
    plt.savefig('./random_walk.png')
    plt.close()
