import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from abc import ABC, abstractmethod


class RandomWalk:
    '''
    Random walk env
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


    def get_grad(self, state):
        pass


    def get_feature_vector(self, state):
        pass


    @abstractmethod
    def learn(self, state, delta):
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
        '''
        Get value of state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        value: float
            value of the current state
        '''
        value = self.w[state]
        return value


    def learn(self, state, error):
        '''
        Update weight vector @self.w

        Params
        ------
        delta: error
        '''
        self.w[state] += error


class TDLambdaValueFunction(ValueFunction):

    def __init__(self, n_states):
        super().__init__(n_states)


    def get_value(self, state):
        '''
        Get value of state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        value: float
            value of the current state
        '''
        value = self.w[state]
        return value


    def get_feature_vector(self, state):
        '''
        Get feature vector of state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        feature_vector: np.ndarray
            feature vector of the current state
        '''
        feature_vector = np.zeros(self.w.shape)
        feature_vector[state] = 1
        return feature_vector


    def get_grad(self, state):
        '''
        Get gradient w.r.t @self.w at state @state
        which is the feature  vector @self.x since 
        using linear func approx

        Params
        ------
        state: int
            current state

        Return
        ------
        grad: np.ndarray
            gradient
        '''
        feature_vector = self.get_feature_vector(state)
        grad = feature_vector
        return grad


    def learn(self, error):
        '''
        Update weight vector @self.w

        Params
        ------
        delta: float
        '''
        self.w += error


class TrueOnlineTDLambdaValueFunction(ValueFunction):

    def __init__(self, n_states):
        super().__init__(n_states)


    def get_value(self, state):
        '''
        Get value of state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        value: float
            value of the current state
        '''
        value = self.w[state]
        return value


    def get_feature_vector(self, state):
        '''
        Get feature vector of state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        feature_vector: np.ndarray
            feature vector of the current state
        '''
        feature_vector = np.zeros(self.w.shape)
        feature_vector[state] = 1
        return feature_vector


    def get_grad(self, state):
        '''
        Get gradient w.r.t @self.w at state @state
        which is the feature  vector @self.x since 
        using linear func approx

        Params
        ------
        state: int
            current state

        Return
        ------
        grad: np.ndarray
            gradient
        '''
        feature_vector = self.get_feature_vector(state)
        grad = feature_vector
        return grad


    def learn(self, error):
        '''
        Update weight vector @self.w

        Params
        ------
        error: float
        '''
        self.w += error


def get_true_value(random_walk, gamma):
    '''
    Calculate true value of @random_walk by Bellman equations

    Params
    ------
    random_walk: RandomWalk
    gamma: float
        discount factor

    Return
    ------
    true_value: np.ndarray
        true value off all of the states
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
    Policy choosing actions randomly

    Params
    ------
    random_walk: RandomWalk

    Return
    ------
    action: int
        chosen action
    '''
    action = np.random.choice(random_walk.actions)
    return action


def offline_lambda_return(value_function, lambda_, alpha, gamma, random_walk):
    '''
    Offline lambda-return algorithm

    Params
    ------
    value_function: ValueFunction
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
                value_function.learn(states[t], delta)
            break


def td_lambda(value_function, lambda_, alpha, gamma, random_walk):
    '''
    TD(lambda) algorithm

    Params
    ------
    value_function: ValueFunction
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
    eligible_trace = np.zeros(value_function.w.shape)

    while not random_walk.is_terminal(state):
        action = random_policy(random_walk)
        next_state, reward = random_walk.take_action(state, action)
        eligible_trace = gamma * lambda_ * eligible_trace + value_function.get_grad(state)
        td_error = reward + gamma * value_function.get_value(next_state) - value_function.get_value(state)
        delta = alpha * td_error * eligible_trace
        value_function.learn(delta)
        state = next_state


def true_online_td_lambda(value_function, lambda_, alpha, gamma, random_walk):
    '''
    True online TD(lambda) algorithm

    Params
    ------
    value_function: ValueFunction
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
    
    dutch_trace = np.zeros(value_function.w.shape)
    zero_vector = np.zeros(value_function.w.shape)
    old_state_value = 0

    while not random_walk.is_terminal(state):
        action = random_policy(random_walk)
        next_state, reward = random_walk.take_action(state, action)
        state_value = value_function.get_value(state)
        state_feature_vector = value_function.get_feature_vector(state)
        next_state_value = value_function.get_value(next_state)
        td_error = reward + gamma * next_state_value - state_value
        dutch_trace = gamma * lambda_ * dutch_trace + (1 - alpha * gamma * lambda_ \
            * dutch_trace.dot(state_feature_vector)) * state_feature_vector
        delta = alpha * ((td_error + state_value - old_state_value) * dutch_trace \
            - (state_value - old_state_value) * state_feature_vector)
        value_function.learn(delta)
        state = next_state
        old_state_value = next_state_value


if __name__ == '__main__':
    n_states = 19
    start_state = 10
    gamma = 1
    random_walk = RandomWalk(n_states, start_state)
    true_value = get_true_value(random_walk, gamma)

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
            'func': offline_lambda_return,
            'value_function': LambdaReturnValueFunction,
            'step_sizes': offline_lambd_return_alphas,
            'img_path': './offline-lambda-return.png'
        },
        {
            'func': td_lambda,
            'value_function': TDLambdaValueFunction,
            'step_sizes': td_lambda_alphas,
            'img_path': './td-lambda.png'
        },
        {
            'func': true_online_td_lambda,
            'value_function': TrueOnlineTDLambdaValueFunction,
            'step_sizes': true_online_td_lambda_alphas,
            'img_path': './true-online-td-lambda.png'
        }
    ]

    errors = []

    for method_idx in range(len(methods)):
        func = methods[method_idx]['func']
        value_func = methods[method_idx]['value_function']
        alphas = methods[method_idx]['step_sizes']

        error = [np.zeros(len(alphas_)) for alphas_ in alphas]

        for _ in trange(runs):
            for lambda_idx in range(len(lambdas)):
                for alpha_idx, alpha in enumerate(alphas[lambda_idx]):
                    value_function = value_func(n_states)

                    for ep in range(episodes):
                        func(value_function, lambdas[lambda_idx], alpha, gamma, random_walk)
                        values = [value_function.get_value(state) for state in random_walk.states]
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
