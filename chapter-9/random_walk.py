import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from env import TransitionRadiusRandomWalk as RandomWalk


def get_true_value(env: RandomWalk) -> np.ndarray:
    '''
    Calculate true values of @env by Dynamic programming

    Params
    ------
    env: RandomWalk env

    Return
    ------
    true_value: true values of @env's states
    '''
    # With this random walk, it makes sense to initialize the values 
    # in the closed interval [-1, 1] and increasing
    n_states = env.n_states
    true_value = np.arange(-(n_states + 1), n_states + 3, 2) / (n_states + 1)
    theta = 1e-2

    while True:
        old_value = true_value.copy()
        for state in env.state_space:
            true_value[state] = 0
            trajectory = []

            for action in env.action_space:
                state_transition = env.get_state_transition(state, action)

                for next_state in state_transition:
                    state_trans_prob = state_transition[next_state]
                    true_value[state] += env.transition_probs[action] \
                        * state_trans_prob * true_value[next_state]

        delta = np.sum(np.abs(old_value - true_value))
        if delta < theta:
            break

    true_value[0] = true_value[-1] = 0

    return true_value


class ValueFunction(ABC):


    def __init__(self):
        pass


    @abstractmethod
    def get_value(self, state: int, terminated: bool=None) -> float:
        '''
        Get value of the state @state

        Params
        ------
        state: state of the agent
        terminated: whether @state is a terminal state
        '''
        pass


    @abstractmethod
    def update(self, state: int, error: float) -> None:
        '''
        Update weight vector

        Params
        ------
        state: state of the agent
        error: update amount
        '''
        pass


class StateAggregationValueFunction(ValueFunction):
    '''
    Value Function using state aggreagation as feature mapping
    '''

    def __init__(self, n_groups: int, n_states: int) -> None:
        '''
        Params
        ------
        n_groups: number of groups
        n_states: number of states
        '''
        self.n_groups = n_groups
        self.group_size = n_states // n_groups
        self.weights = np.zeros(n_groups)


    def _get_group(self, state: int) -> int:
        '''
        Get group index

        Params
        ------
        state: state of the agent

        Return
        ------
        group_idx: group index
        '''
        group_idx = (state - 1) // self.group_size
        return group_idx


    def _get_grad(self, state: int) -> np.ndarray:
        '''
        Compute the gradient w.r.t @self.weights at state @state

        Params
        ------
        state: state of the agent

        Return
        ------
        grad: gradient w.r.t @self.weights at the state @state
        '''
        group_idx = self._get_group(state)
        grad = np.zeros(self.n_groups)
        grad[group_idx] = 1
        return grad


    def get_value(self, state: int, terminated: bool=None) -> float:
        '''
        Get value of state @state
        States within a group share the same value function,
        which is a component of @self.weights

        Params
        ------
        state: state of the agent
        terminated: whether state @state is terminal

        Return
        ------
        value: value of state @state
        '''
        if terminated is not None and terminated:
            value = 0
        else:
            group_idx = self._get_group(state)
            value = self.weights[group_idx]
        return value


    def update(self, state: int, error: float) -> None:
        '''
        Update weight vector

        Params
        ------
        state: state of the agent
        error: update amount
        '''
        grad = self._get_grad(state)
        self.weights += error * grad


Feature_mapping = Callable[[int, int], float]

class BasesValueFunction(ValueFunction):
    '''
    Value function using polynomial/Fourier basis as feature mapping
    '''

    def __init__(self, order: int, basis_type: str, 
                n_states: int) -> None:
        '''
        Params
        ------
        order: order
        basis_type: basis type
        n_states: number of states
        '''
        self.order = order
        self.basis_type = basis_type
        self.basis_types = ['Polynomial', 'Fourier']
        self.n_states = n_states
        # additional basis for bias
        self.weights = np.zeros(order + 1)
        self.features = self._get_features()


    def _get_features(self) -> List[Feature_mapping]:
        '''
        Get feature vector functions with 1-dim state

        Return
        ------
        features: list of feature mapping functions
        '''
        features = []
        if self.basis_type == self.basis_types[0]:
            for i in range(self.order + 1):
                features.append(lambda s, i=i: pow(s, i))
        elif self.basis_type == self.basis_types[1]:
            for i in range(self.order + 1):
                features.append(lambda s, i=i: np.cos(i * np.pi * s))
        return features


    def _get_feature_vector(self, state: int) -> np.ndarray:
        '''
        Get feature vector of state @state

        Params
        ------
        state: state of the agent

        Return
        ------
        feature_vector: feature vector of the state @state
        '''
        feature_vector = np.asarray([x_i(state) for x_i in self.features])
        return feature_vector


    def _get_grad(self, state: int) -> np.ndarray:
        '''
        Compute the gradient w.r.t @self.w at state @state
        Since value function is approximated by a linear function,
        its gradient w.r.t the weight @self.weights is equal to 
        the feature vector @self.features

        Params
        ------
        state: state of the agent

        Return
        ------
        grad: gradient w.r.t @self.weights at the state @state
        '''
        state /= float(self.n_states)
        feature_vector = self._get_feature_vector(state)
        grad = feature_vector
        return grad


    def get_value(self, state: int, terminated: bool=None) -> float:
        '''
        Get value of the state @state
        value function is equal to dot product of its feature 
        vector and weight corresponding

        Params
        ------
        state: state of the agent
        terminated: whether @state is terminal

        Return
        ------
        value: value of the state @state
        '''
        if terminated is not None and terminated:
            value = 0
        else:
            state /= float(self.n_states)
            feature_vector = self._get_feature_vector(state)
            value = np.dot(self.weights, feature_vector)
        return value


    def update(self, state: int, error: int) -> None:
        '''
        Update weight vector

        Params
        ------
        state: state of the agent
        error: update amount
        '''
        grad = self._get_grad(state)
        self.weights += error * grad


class TilingValueFunction(ValueFunction):

    def __init__(self, n_tilings: int, tile_width: int, 
                tiling_offset: int, n_states: int) -> None:
        '''
        Params:
        ------
        n_tilings: number of tilings
        tile_width: tile width
        tiling_offset: tiling offset
        n_state: number of states
        '''
        self.n_tilings = n_tilings
        self.tile_width = tile_width
        self.tiling_offset = tiling_offset

        # we need 1 more tile for each tiling to make sure that 
        # each state is covered by the same number of tiles
        # within an interval with length = @self.tiling_size, all
        # states activate the same tiles, have the same feature 
        # representation, and therefore the same value function.
        self.tiling_size = n_states // tile_width + 1
        self.weights = np.zeros((n_tilings, self.tiling_size))


    def _get_active_tiles(self, state: int) -> List[int]:
        '''
        Get list of (indices of) active tiles

        Params
        ------
        state: state of the agent

        Return
        ------
        active_tiles: list of (indices of) active tiles
        '''
        active_tiles = []

        for tiling_idx in range(self.n_tilings):
            tile_idx = (state - self.tiling_offset * tiling_idx - 1) \
                    // self.tile_width + 1
            active_tiles.append(tile_idx)
                
        return active_tiles


    def get_value(self, state: int) -> float:
        '''
        Get value of the state @state

        Params
        ------
        state: state of the agent

        Return
        ------
        value: value of the state @state
        '''
        value = 0
        active_tiles = self._get_active_tiles(state)
        for tiling_idx, tile_idx in enumerate(active_tiles):
            value += self.weights[tiling_idx, tile_idx]

        return value


    def update(self, state: int, error: float) -> None:
        '''
        Update weight vector

        Params
        ------
        state: state of the agent
        error: update amount
        '''
        active_tiles = self._get_active_tiles(state)
        error /= self.n_tilings

        for tiling_idx in range(self.n_tilings):
            self.weights[tiling_idx, active_tiles[tiling_idx]] += error


class Agent(ABC):
    '''
    Agent abstract class
    '''

    def __init__(self, env: RandomWalk, 
                value_function: ValueFunction,
                alpha: float, gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        value_function: value function
        alpha: step size param
        gamma : discount factor
        '''
        self.env = env
        self.value_function = value_function
        self.alpha = alpha
        self.gamma = gamma


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
    def learn(self) -> None:
        '''
        Update weights vector by SGD method
        '''
        pass


    @abstractmethod
    def run(self) -> None:
        '''
        Perform an episode
        '''
        pass


class GradientMonteCarlo(Agent):
    '''
    Gradient Monte Carlo agent
    '''

    def __init__(self, env: RandomWalk,
                value_function: ValueFunction,
                alpha: float, gamma: float,
                mu: np.ndarray=None) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        value_function: value function
        alpha: step size param
        gamma : discount factor
        mu: state distribution
        '''
        super().__init__(env, value_function, alpha, gamma)
        self.mu = mu


    def learn(self, state: int, target: float, estimate: float) -> None:
        '''
        Update weight vector by SGD method

        Params
        ------
        state: state of the agent
        target: target of the update
        estimate: estimate of the update
        '''
        error = target - estimate
        error *= self.alpha
        self.value_function.update(state, error)


    def run(self) -> None:
        '''
        Perform an episode
        '''
        self.reset()
        trajectory = []

        while True:
            action = self.random_policy()
            state = self.env.state
            next_state, reward, terminated = self.env.step(action)
            for t in range(len(trajectory)):
                trajectory[t][1] += np.power(self.gamma, len(trajectory) - t) * reward
            trajectory.append([state, reward])

            if terminated:
                break

        for state, return_ in trajectory:
            self.learn(state, return_, self.value_function.get_value(state))
            if self.mu is not None:
                self.mu[state] += 1


class NStepSemiGradientTD(Agent):
    '''
    n-step semi-gradient TD agent
    '''

    def __init__(self, env: RandomWalk,
                value_function: ValueFunction,
                n: int, alpha: float, gamma: float) -> None:
        '''
        Params
        ------
        env: RandomWalk env
        value_function: value function
        n: number of steps
        alpha: step size param
        gamma : discount factor
        '''
        super().__init__(env, value_function, alpha, gamma)
        self.n = n


    def learn(self, state: int, target: float, estimate: float) -> None:
        '''
        Update weight vector by SGD method

        Params
        ------
        state: state of the agent
        target: target of the update
        estimate: estimate of the update
        '''
        error = target - estimate
        error *= self.alpha
        self.value_function.update(state, error)


    def run(self) -> None:
        '''
        Perform an episode
        '''
        self.reset()
        states = [self.env.state]
        rewards = [0] # dummy reward to save the next reward as R_{t+1}
        terminates = [False] # flag list to indicate whether S_t is terminal
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
            tau = t - self.n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += np.power(self.gamma, i - tau - 1) * rewards[i]
                if tau + self.n < T:
                    G += np.power(self.gamma, self.n) * self.value_function.get_value(
                        states[tau + self.n], terminates[tau + self.n])
                if not terminates[tau]:
                    self.learn(states[tau], G, 
                        self.value_function.get_value(states[tau]))
            t += 1
            if tau == T - 1:
                break


def gradient_mc_state_aggregation_plot(env: RandomWalk,
                        true_value: np.ndarray) -> None:
    '''
    Plot gradient MC w/ state aggregation

    Params
    ------
    env: RandomWalk env
    true_value: true values
    '''
    alpha = 2e-5
    gamma = 1
    n_groups = 10
    n_eps = 100000
    mu = np.zeros(env.n_states + 2)
    value_function = StateAggregationValueFunction(n_groups, env.n_states)
    gradient_mc = GradientMonteCarlo(env, value_function, alpha, gamma, mu)

    for _ in trange(n_eps):
        gradient_mc.run()

    mu /= np.sum(mu)
    values = [value_function.get_value(state) for state in env.state_space]

    fig, ax1 = plt.subplots()
    value_func_plot = ax1.plot(env.state_space, values, 
        label=r'Approximate MC value $\hat{v}$', color='blue')
    true_value_plot = ax1.plot(env.state_space, true_value[1: -1], 
        label=r'True value $v_\pi$', color='red')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value scale')
    ax2 = ax1.twinx()
    state_dist_plot = ax2.plot(env.state_space, mu[1: -1], 
        label=r'State distribution $\mu$', color='gray')
    ax2.set_ylabel('Distribution scale')
    plots = value_func_plot + true_value_plot + state_dist_plot
    labels = [l.get_label() for l in plots]
    plt.legend(plots, labels, loc=0)
    plt.savefig('./gradient_mc_state_agg.png')
    plt.close()


def semi_gradient_td_0_plot(env: RandomWalk, 
                        true_value: np.ndarray) -> None:
    '''
    Plot semi-gradient TD(0)

    Params
    ------
    env: RandomWalk env
    true_value: true values
    '''
    alpha = 2e-4
    n_groups = 10
    n_eps = 100000
    gamma = 1
    value_function = StateAggregationValueFunction(n_groups, env.n_states)
    semi_grad_td_0 = NStepSemiGradientTD(env, value_function, 1, alpha, gamma)

    for _ in trange(n_eps):
        semi_grad_td_0.run()

    values = [value_function.get_value(state) for state in env.state_space]

    plt.plot(env.state_space, values, 
        label=r'Approximate TD value $\hat{v}$', color='blue')
    plt.plot(env.state_space, true_value[1: -1], 
        label=r'True value $v_\pi$', color='red')
    plt.xlabel('State')
    plt.ylabel('Value scale')
    plt.legend()


def n_step_semi_gradient_td_plot(env: RandomWalk,
                            true_value: np.ndarray) -> None:
    '''
    Plot n-step semi-gradient TD

    Params
    ------
    env: RandomWalk env
    true_value: true values
    '''
    n_eps = 10
    n_runs = 100
    n_groups = 20
    gamma = 1
    ns = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)

    errors = np.zeros((len(ns), len(alphas)))
    for n_i, n in enumerate(ns):
        for alpha_i, alpha in enumerate(alphas):
            print(f'n={n}, alpha={alpha}')
            for _ in trange(n_runs):
                value_function = StateAggregationValueFunction(n_groups, env.n_states)
                n_step_semi_grad_td = NStepSemiGradientTD(env, value_function, n, alpha, gamma)

                for _ in range(n_eps):
                    n_step_semi_grad_td.run()
                    values = np.array([value_function.get_value(state)
                        for state in env.state_space])
                    rmse = np.sqrt(np.sum(np.power(values - true_value[1: -1], 2) 
                        / env.n_states))
                    errors[n_i, alpha_i] += rmse

    errors /= n_eps * n_runs

    for i in range(0, len(ns)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (ns[i]))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()


def semi_gradient_td_plot(env: RandomWalk,
                    true_value: np.ndarray) -> None:
    '''
    Plot Semi-gradient TD methods

    Params
    ------
    env: RandomWalk env
    true_value: true values
    '''
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    semi_gradient_td_0_plot(env, true_value)
    plt.subplot(122)
    n_step_semi_gradient_td_plot(env, true_value)
    plt.savefig('./semi_gradient_td.png')
    plt.close()


def gradient_mc_tilings_plot(env: RandomWalk,
                    true_value: np.ndarray) -> None:
    '''
    Plot gradient Monte Carlo w/ single and multiple tilings
    The single tiling method is basically state aggregation.

    Params
    ------
    env: RandomWalk env
    true_value: true values
    '''
    n_runs = 1
    n_eps = 5000
    n_tilings = 50
    tile_width = 200
    tiling_offset = 4
    gamma = 1

    plot_labels = ['state aggregation (one tiling)', 'tile coding (50 tilings)']

    errors = np.zeros((len(plot_labels), n_eps))

    for _ in range(n_runs):
        value_functions = [
            StateAggregationValueFunction(env.n_states // tile_width, env.n_states), 
            TilingValueFunction(n_tilings, tile_width, tiling_offset, env.n_states)
        ]

        for i in range(len(value_functions)):

            for ep in trange(n_eps):
                alpha = 1.0 / (ep + 1)
                gradient_mc = GradientMonteCarlo(env, value_functions[i], alpha, gamma)
                gradient_mc.run()
                values = [value_functions[i].get_value(state) for state in env.state_space]
                errors[i][ep] += np.sqrt(np.mean(np.power(true_value[1: -1] - values, 2)))

    errors /= n_runs
    for i in range(len(plot_labels)):
        plt.plot(errors[i], label=plot_labels[i])
    plt.xlabel('Episodes')
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('./gradient_mc_tile_coding.png')
    plt.close()


def gradient_mc_bases_plot(env: RandomWalk,
                    true_value: np.ndarray) -> None:
    '''
    Plot gradient Monte Carlo w/ Fourier and polynomial bases

    Params
    ------
    env: RandomWalk env
    true_value: true values
    '''
    orders = [5, 10, 20]
    n_runs = 1
    n_eps = 5000
    gamma = 1

    bases = [
        {'method': 'Polynomial', 'alpha': 1e-4},
        {'method': 'Fourier', 'alpha': 5e-5}
    ]

    errors = np.zeros((len(bases), len(orders), n_eps))
    for i_basis, basis in enumerate(bases):
        for i_order, order in enumerate(orders):
            print(f'{basis["method"]} basis, order={order}')
            for _ in range(n_runs):
                value_function = BasesValueFunction(order, basis['method'], env.n_states)
                gradient_mc = GradientMonteCarlo(env, value_function, basis['alpha'], gamma)

                for ep in trange(n_eps):
                    gradient_mc.run()
                    values = np.array([value_function.get_value(state) 
                        for state in env.state_space])
                    rmse = np.sqrt(np.mean(np.power(values - true_value[1: -1], 2)))
                    errors[i_basis, i_order, ep] += rmse

    errors /= n_runs
    for i_basis, basis in enumerate(bases):
        for i_order, order in enumerate(orders):
            plt.plot(errors[i_basis, i_order, :], label='%s basis, order = %d' \
                % (basis['method'], order))
    plt.xlabel('Episodes')
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('./gradient_mc_bases.png')
    plt.close()


if __name__ == '__main__':
    n_states = 1000
    start_state = 500
    terminal_states = [0, n_states + 1]
    transition_radius = 100
    env = RandomWalk(n_states, start_state, terminal_states,
        transition_radius=transition_radius)
    true_value = get_true_value(env)

    gradient_mc_state_aggregation_plot(env, true_value)
    semi_gradient_td_plot(env, true_value)
    gradient_mc_tilings_plot(env, true_value)
    gradient_mc_bases_plot(env, true_value)
