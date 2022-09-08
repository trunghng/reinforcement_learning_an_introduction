import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from abc import ABC, abstractmethod

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


class EligibleTraceAgent(ABC):
    '''
    Agent abstract class
    '''

    def __init__(self, env: RandomWalk, 
                lambda_: float, alpha: float,
                gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        lambda_: trace decay param
        alpha: step size param
        gamma: discount factor
        '''
        self.env = env
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.weights = np.zeros(env.n_states + 2)


    @abstractmethod
    def __call__(self, env: RandomWalk, 
                lambda_: float, alpha: float,
                gamma: float) -> object:
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


    def get_value(self, state: int) -> float:
        '''
        Get value of state @state

        Params
        ------
        state: state of the agent

        Return
        ------
        value: state value
        '''
        value = self.weights[state]
        return value


    def get_feature_vector(self, state: int) -> np.ndarray:
        '''
        Get feature vector of state @state

        Params
        ------
        state: state of the agent

        Return
        ------
        feature_vector: feature vector corresponding to @state
        '''
        feature_vector = np.zeros(self.weights.shape)
        feature_vector[state] = 1
        return feature_vector


    def get_grad(self, state: int) -> np.ndarray:
        '''
        Get gradient w.r.t @self.w at state @state
        which is the feature  vector @self.x since 
        using linear func approx

        Params
        ------
        state: state of the agent

        Return
        ------
        grad: gradient vector corresponding to @state
        '''
        feature_vector = self.get_feature_vector(state)
        grad = feature_vector
        return grad


    @abstractmethod
    def learn(self) -> None:
        pass


    @abstractmethod
    def run(self) -> None:
        pass


class OfflineLambdaReturn(EligibleTraceAgent):
    '''
    Offline Lambda-return agent
    '''

    def __init__(self, env: RandomWalk,
                lambda_: float, alpha: float,
                gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        lambda_: trace decay param
        alpha: step size param
        gamma: discount factor
        '''
        super().__init__(env, lambda_, alpha, gamma)
        self.lambda_truncate = 1e-3


    def __call__(self, env: RandomWalk,
                lambda_: float, alpha: float,
                gamma: float) -> object:
        return OfflineLambdaReturn(env, lambda_, alpha, gamma)


    def learn(self, state: int, error: float) -> None:
        '''
        Update weight vector by SGD method

        Params
        ------
        state: state of the agent
        error: update amount
        '''
        self.weights[state] += error


    def run(self) -> None:
        '''
        Perform an episode
        '''
        start_state = self.reset()
        states = [start_state]

        while True:
            action = self.random_policy()
            next_state, reward, terminated = self.env.step(action)
            states.append(next_state)

            if terminated:
                T = len(states) - 1

                for t in range(T):
                    lambda_return = 0

                    for n in range(1, T - t):
                        n_step_return = np.power(self.gamma, t + n) \
                            * self.get_value(states[t + n])
                        lambda_return += np.power(self.lambda_, t + n - 1) * n_step_return
                        if np.power(self.lambda_, t + n - 1) < self.lambda_truncate:
                            break

                    lambda_return *= 1 - self.lambda_
                    if np.power(self.lambda_, T - t - 1) >= self.lambda_truncate:
                        lambda_return += np.power(self.lambda_, T - t - 1) * reward
                    error = self.alpha * (lambda_return - self.get_value(states[t]))
                    self.learn(states[t], error)
                break


class TDLambda(EligibleTraceAgent):
    '''
    TD(lambda) agent
    '''

    def __init__(self, env: RandomWalk,
                lambda_: float, alpha: float,
                gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        lambda_: trace decay param
        alpha: step size param
        gamma: discount factor
        '''
        super().__init__(env, lambda_, alpha, gamma)


    def __call__(self, env: RandomWalk,
                lambda_: float, alpha: float,
                gamma: float) -> object:
        return TDLambda(env, lambda_, alpha, gamma)


    def learn(self, error: float) -> None:
        '''
        Update weight vector by SGD method

        Params
        ------
        error: update amount
        '''
        self.weights += error


    def run(self) -> None:
        '''
        Perform an episode
        '''
        self.reset()
        eligible_trace = np.zeros(self.weights.shape)

        while True:
            action = self.random_policy()
            next_state, reward, terminated = self.env.step(action)
            eligible_trace = self.gamma * self.lambda_ * eligible_trace \
                + self.get_grad(self.env.state)
            td_error = reward + gamma * self.get_value(next_state) \
                - self.get_value(self.env.state)
            error = self.alpha * td_error * eligible_trace
            self.learn(error)

            if terminated:
                break


class TrueOnlineTDLambda(EligibleTraceAgent):
    '''
    True online TD(lambda) agent
    '''

    def __init__(self, env: RandomWalk,
                lambda_: float, alpha: float,
                gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        lambda_: trace decay param
        alpha: step size param
        gamma: discount factor
        '''
        super().__init__(env, lambda_, alpha, gamma)


    def __call__(self, env: RandomWalk,
                lambda_: float, alpha: float,
                gamma: float) -> object:
        return TrueOnlineTDLambda(env, lambda_, alpha, gamma)


    def learn(self, error: float) -> None:
        '''
        Update weight vector by SGD method

        Params
        ------
        error: update amount
        '''
        self.weights += error


    def run(self) -> None:
        '''
        Perform an episode
        '''
        self.reset()
        dutch_trace = np.zeros(self.weights.shape)
        zero_vector = np.zeros(self.weights.shape)
        old_state_value = 0

        while True:
            action = self.random_policy()
            next_state, reward, terminated = self.env.step(action)
            state_value = self.get_value(self.env.state)
            state_feature_vector = self.get_feature_vector(self.env.state)
            next_state_value = self.get_value(next_state)
            td_error = reward + self.gamma * next_state_value - state_value
            dutch_trace = self.gamma * self.lambda_ * dutch_trace \
                + (1 - self.alpha * self.gamma * self.lambda_ \
                * dutch_trace.dot(state_feature_vector)) * state_feature_vector
            error = self.alpha * ((td_error + state_value - old_state_value) 
                * dutch_trace - (state_value - old_state_value) * state_feature_vector)
            self.learn(error)
            old_state_value = next_state_value

            if terminated:
                break


if __name__ == '__main__':
    n_states = 19
    start_state = 10
    terminal_states = [0, n_states + 1]
    env = RandomWalk(n_states, start_state ,terminal_states)
    gamma = 1
    true_value = get_true_value(env, gamma)

    episodes = 10
    runs = 50
    lambdas = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    offline_lambd_return_alphas = [
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 0.55, 0.05),
        np.arange(0, 0.22, 0.02),
        np.arange(0, 0.11, 0.01)
    ]
    td_lambda_alphas = [
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 0.99, 0.09),
        np.arange(0, 0.55, 0.05),
        np.arange(0, 0.33, 0.03),
        np.arange(0, 0.22, 0.02),
        np.arange(0, 0.11, 0.01),
        np.arange(0, 0.044, 0.004)
    ]
    true_online_td_lambda_alphas = [
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        np.arange(0, 0.88, 0.08),
        np.arange(0, 0.44, 0.04),
        np.arange(0, 0.11, 0.01)
    ]

    methods = [
        {
            'agent': OfflineLambdaReturn,
            'step_sizes': offline_lambd_return_alphas,
            'img_path': './random-walk-offline-lambda-return.png'
        },
        {
            'agent': TDLambda,
            'step_sizes': td_lambda_alphas,
            'img_path': './random-walk-td-lambda.png'
        },
        {
            'agent': TrueOnlineTDLambda,
            'step_sizes': true_online_td_lambda_alphas,
            'img_path': './random-walk-true-online-td-lambda.png'
        }
    ]

    errors = []

    for method_idx in range(len(methods)):
        agent = methods[method_idx]['agent']
        alphas = methods[method_idx]['step_sizes']

        error = [np.zeros(len(alphas_)) for alphas_ in alphas]

        for _ in trange(runs):
            for lambda_idx in range(len(lambdas)):
                for alpha_idx, alpha in enumerate(alphas[lambda_idx]):
                    agent = agent(env, lambdas[lambda_idx], alpha, gamma)

                    for ep in range(episodes):
                        agent.run()
                        values = [agent.get_value(state) for state in env.state_space]
                        error[lambda_idx][alpha_idx] += np.sqrt(np.mean(np.power
                            (values - true_value[1: -1], 2)))

        errors.append(error)

    for errors_ in errors:
        for error in errors_:
            error /= episodes * runs

    for method_idx in range(len(methods)):
        for lambda_idx in range(len(lambdas)):
            plt.plot(alphas[lambda_idx], errors[method_idx][lambda_idx], 
                label= r'$\lambda$ = ' + str(lambdas[lambda_idx]))
        plt.xlabel('alpha')
        plt.ylabel('RMS error')
        plt.legend(loc='upper right')
        plt.savefig(methods[method_idx]['img_path'])
    plt.close()
