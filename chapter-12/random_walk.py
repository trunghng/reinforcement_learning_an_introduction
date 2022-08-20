import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from abc import ABC, abstractmethod


class RandomWalk:
    '''
    Random walk environment
    '''

    def __init__(self, n_states, start_state):
        self.n_states = n_states
        self.states = np.arange(1, n_states + 1)
        self.start_state = start_state
        self.end_states = [0, n_states + 1]
        self.actions = [-1, 1]
        self.action_prob = 0.5
        self.rewards = [-1, 0, 1]


    def is_terminal(self, state):
        '''
        Whether state @state is an end state

        Params
        ------
        state: int
            current state
        '''
        return state in self.end_states


    def get_reward(self, next_state):
        '''
        Get reward corresponding to the next state

        Params
        ------
        next_state: int
            next state

        Return
        ------
        reward: int
            next reward
        '''
        if next_state == 0:
            reward = self.rewards[0]
        elif next_state == self.n_states + 1:
            reward = self.rewards[2]
        else:
            reward = self.rewards[1]
        return reward


    def take_action(self, state, action):
        '''
        Take action @action at state @state

        Params
        ------
        state: int
            current state
        action: int
            action taken

        Return
        ------
        (next_state, reward): (int, int)
            a tuple of next state and reward
        '''
        next_state = state + action
        reward = self.get_reward(next_state)
        return next_state, reward


class ValueFunction(ABC):
    '''
    value function abstract class
    '''

    def __init__(self, n_states):
        self.w = np.zeros(n_states + 2)


    @abstractmethod
    def get_value(self, state):
        pass


    @abstractmethod
    def update_weights(self, state, delta):
        pass


class LambdaReturnValueFunction(ValueFunction):
    '''
    We use state aggregation as the feature constructor and with 
    a small state space (19 states), each state represents for a group.
    Hence, the value of each state = the corresponding weight
    '''

    def __init__(self, n_states):
        super().__init__(n_states)


    def get_value(self, state):
        return self.w[state]


    def update_weights(self, state, delta):
        self.w[state] += delta


class TDLambdaValueFunction(ValueFunction):

    def __init__(self, n_states):
        super().__init__(n_states)


    def get_value(self, state):
        return self.w[state]


    def update_weights(self, delta):
        self.w += delta


def get_true_value(random_walk, gamma):
    '''
    Calculate true value of @random_walk by Bellman equations

    Params
    ------
    random_walk: RandomWalk
    gamma: float
        discount factor
    '''
    P = np.zeros((random_walk.n_states, random_walk.n_states))
    r = np.zeros((random_walk.n_states + 2, ))
    true_value = np.zeros((random_walk.n_states + 2, ))
    
    for state in random_walk.states:
        next_states = []
        rewards = []

        for action in random_walk.actions:
            next_state = state + action
            next_states.append(next_state)
            reward = random_walk.get_reward(next_state)
            rewards.append(reward)

        for state_, reward_ in zip(next_states, rewards):
            if not random_walk.is_terminal(state_):
                P[state - 1, state_ - 1] = random_walk.action_prob * 1
                r[state_] = reward_
        
    u = np.zeros((random_walk.n_states, ))
    u[0] = random_walk.action_prob * 1 * (-1 + gamma * random_walk.rewards[0])
    u[-1] = random_walk.action_prob * 1 * (1 + gamma * random_walk.rewards[2])

    r = r[1:-1]
    true_value[1:-1] = np.linalg.inv(np.identity(random_walk.n_states) - gamma * P).dot(0.5 * (P.dot(r) + u))
    true_value[0] = true_value[-1] = 0

    return true_value


def random_policy(random_walk):
    '''
    Choose an action randomly

    Params
    ------
    random_walk: RandomWalk
    '''
    return np.random.choice(random_walk.actions)


def offline_lambda_return(value_function, lambda_, alpha, gamma, random_walk):
    '''
    Offline lambda-return

    Params
    ------
    value_function: np.ndarray
        value function
    lambda_: float
        trace decay param
    alpha: float
        step size
    gamma: float
        discount factor
    random_walk: RandomWalk
    '''
    state = random_walk.start_state
    states = [state]
    lambda_truncate = 1e-3

    while True:
        action = random_policy(random_walk)
        next_state, reward = random_walk.take_action(state, action)
        state = next_state
        states.append(state)
        if random_walk.is_terminal(state):
            T = len(states) - 1

            for t in range(T):
                lambda_return = 0

                for n in range(1, T - t):
                    n_step_return = np.power(gamma, t + n) * value_function.get_value(states[t + n])
                    lambda_return += np.power(lambda_, t + n - 1) * n_step_return
                    if np.power(lambda_, t + n - 1) < lambda_truncate:
                        break

                lambda_return *= 1 - lambda_
                if np.power(lambda_, T - t - 1) >= lambda_truncate:
                    lambda_return += np.power(lambda_, T - t - 1) * reward
                delta = alpha * (lambda_return - value_function.get_value(states[t]))
                value_function.update_weights(states[t], delta)
            break


def online_lambda_return(value_function, lambda_, alpha, gamma, random_walk):
    '''
    Online lambda-return

    Params
    ------
    value_function: np.ndarray
        value function
    lambda_: float
        trace decay param
    alpha: float
        step size
    gamma: float
        discount factor
    random_walk: RandomWalk
    '''
    state = random_walk.start_state
    eligible_trace = np.zeros(random_walk.n_states + 2)

    while True:
        action = random_policy(random_walk)
        next_state, reward = random_walk.take_action(state, action)
        eligible_trace = gamma * lambda_ * eligible_trace + value_function.get_value(state)
        td_error = reward + gamma * value_function.get_value(next_state) - value_function.get_value(state)
        delta = alpha * td_error * eligible_trace


def semi_gradient_td_lambda(value_function, lambda_, alpha, gamma, random_walk):
    '''
    Semi-gradient TD(lambda)

    Params
    ------
    value_function: np.ndarray
        value function
    lambda_: float
        trace decay param
    alpha: float
        step size
    gamma: float
        discount factor
    random_walk: RandomWalk
    '''
    pass


if __name__ == '__main__':
    n_states = 19
    start_state = 10
    gamma = 1
    random_walk = RandomWalk(n_states, start_state)
    true_value = get_true_value(random_walk, gamma)

    episodes = 10
    runs = 50
    lambdas = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01)]

    methods = [
        {
            'name': r'Offline $\lambda$-return',
            'bootstrapping_param': lambdas,
            'bootstrapping_param_name': 'lambda',
            'func': offline_lambda_return,
            'value_function': LambdaReturnValueFunction
        }
    ]

    errors = []

    for method_idx in range(len(methods)):
        method_name = methods[method_idx]['name']
        bootstrapping_params = methods[method_idx]['bootstrapping_param']
        bootstrapping_param_name = methods[method_idx]['bootstrapping_param_name']
        func = methods[method_idx]['func']

        # error = np.zeros((len(bootstrapping_params), len(alphas)))
        error = [np.zeros(len(alphas_)) for alphas_ in alphas]

        for _ in trange(runs):
            for param_idx in range(len(bootstrapping_params)):
                for alpha_idx, alpha in enumerate(alphas[param_idx]):
                    print(f'alpha={alpha}, {bootstrapping_param_name}={bootstrapping_params[param_idx]}')   
                    value_function = methods[method_idx]['value_function'](n_states)

                    for ep in range(episodes):
                        func(value_function, bootstrapping_params[param_idx], 
                            alpha, gamma, random_walk)
                        values = [value_function.get_value(state) for state in random_walk.states]
                        error[param_idx][alpha_idx] += np.sqrt(np.mean(np.power
                            (values - true_value[1: -1], 2)))

        errors.append(error)

    for errors_ in errors:
        for error in errors_:
            error /= episodes * runs

    for method_idx in range(len(methods)):
        for param_idx in range(len(methods[method_idx]['bootstrapping_param'])):
            plt.plot(alphas[param_idx], errors[method_idx][param_idx], 
                label=methods[method_idx]['bootstrapping_param_name'] + ' = ' 
                + str(methods[method_idx]['bootstrapping_param'][param_idx]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig('./random_walk.png')
    plt.close()
