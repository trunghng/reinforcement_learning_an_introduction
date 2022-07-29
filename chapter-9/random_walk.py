import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class RandomWalk:
    '''
    Random walk with a transition radius
    '''


    def __init__(self, n_states, start_state, transition_radius):
        self.n_states = n_states
        self.start_state = start_state
        self.transition_radius = transition_radius
        self.states = np.arange(1, n_states + 1)
        self.end_states = [0, n_states + 1]
        self.actions = [-1, 1]
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


    def get_pos_next_states(self, state, action):
        '''
        Get possible states at state @state, taking action @action

        Params
        ------
        state: int
            current state
        action: int
            action taken

        Return
        ------
        pos_next_states: np.ndarray
            list of possible next states
        '''
        if action == self.actions[0]:
            pos_next_states = np.arange(max(self.end_states[0], state - 
                self.transition_radius), state + action + 1)
        else:
            pos_next_states = np.arange(state + action, min(self.end_states[1], 
                state + self.transition_radius) + 1)

        return pos_next_states


    def get_state_transition(self, state, pos_next_states):
        '''
        Get state transition at state @state

        Params
        ------
        state: int
            current state
        pos_next_states: np.ndarray
            list of possible next states

        Return
        ------
        state_transition: np.ndarray
            state transition probability
        '''
        next_state_prob = 1.0 / self.transition_radius
        state_transition = np.array([next_state_prob for _ in pos_next_states])

        if self.end_states[0] == pos_next_states[0]:
            state_transition[0] += (self.transition_radius 
                - len(pos_next_states)) * next_state_prob
        elif self.end_states[1] == pos_next_states[-1]:
            state_transition[-1] += (self.transition_radius 
                - len(pos_next_states)) * next_state_prob

        return state_transition


    def get_next_state(self, state, action):
        step = np.random.randint(1, self.transition_radius + 1)
        next_state = min(self.end_states[1], max(self.end_states[0], state + action * step))
        return next_state


    def get_reward(self, state):
        '''
        Get reward when ending at state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        reward: int
            reward at state @state
        '''
        if state == self.end_states[0]:
            reward = self.rewards[0]
        elif state == self.end_states[1]:
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
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(next_state)

        return next_state, reward


def get_true_value(random_walk):
    '''
    Calculate true values of @random_walk by DP

    Params
    ------
    random_walk: RandomWalk

    Return
    ------
    true_value: np.ndarray
        true values of @random_walk's states
    '''
    # With this random walk, it makes sense to initialize the values in the closed interval 
    # [-1, 1] and increasing
    n_states = random_walk.n_states
    true_value = np.arange(-(n_states + 1), n_states + 3, 2) / (n_states + 1)
    theta = 1e-2

    while True:
        old_value = true_value.copy()
        for idx_state, state in enumerate(random_walk.states):
            true_value[state] = 0

            for idx_action, action in enumerate(random_walk.actions):
                pos_next_states = random_walk.get_pos_next_states(state, action)
                state_transition = random_walk.get_state_transition(state, pos_next_states)

                for i, next_state in enumerate(pos_next_states):
                    trans_prob = state_transition[i]
                    true_value[state] += 0.5 * trans_prob * true_value[next_state]

        delta = np.sum(np.abs(old_value - true_value))
        if delta < theta:
            break

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


class StateAggregationValueFunction:
    '''
    State Aggregation
    '''

    def __init__(self, n_groups, n_states):
        self.n_groups = n_groups
        self.group_size = n_states // n_groups
        self.w = np.zeros(n_groups)


    def get_group(self, state):
        '''
        Get group index

        Params
        ------
        state: int
            current state

        Return
        ------
        group_idx: int
            group index
        '''
        group_idx = (state - 1) // self.group_size
        return group_idx


    def get_grad(self, state):
        '''
        Compute the gradient w.r.t @self.w at state @state

        Params
        ------
        state: int
            current state

        Return
        ------
        grad: np.ndarray
            gradient w.r.t @self.w at the current state
        '''
        group_idx = self.get_group(state)
        grad = np.zeros(self.n_groups)
        grad[group_idx] = 1
        return grad


    def get_value(self, state, random_walk):
        '''
        Get value function at state @state
        States within a group share the same value function, which is a component of w

        Params
        ------
        state: int
            current state
        random_walk: RandomWalk
            random walk

        Return
        ------
        value_func: float
            value function at state @state
        '''
        if random_walk.is_terminal(state):
            value_func = 0
        else:
            group_idx = self.get_group(state)
            value_func = self.w[group_idx]
        return value_func


class BasesValueFunction:
    '''
    State features w/ bases
    '''

    def __init__(self, order, basis_type):
        self.order = order
        # additional basis for bias
        self.w = np.zeros(order + 1)
        self.basis_types = ['Polynomial', 'Fourier']
        self.features = self.get_features(basis_type)


    def get_features(self, basis_type):
        '''
        Get feature vector functions with 1-dim state

        Params:
        -------
        bais_type: str
            basis type

        Return
        ------
        features: list
            list of feature vector functions
        '''
        features = []
        if basis_type == self.basis_types[0]:
            for i in range(self.order + 1):
                features.append(lambda s, i=i: pow(s, i))
        elif basis_type == self.basis_types[1]:
            for i in range(self.order + 1):
                features.append(lambda s, i=i: np.cos(i * np.pi * s))
        return features


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
            feature vector
        '''
        feature_vector = np.asarray([x_i(state) for x_i in self.features])
        return feature_vector


    def get_grad(self, state, random_walk):
        '''
        Compute the gradient w.r.t @self.w at state @state
        Since value function is approximated by a linear function, its gradient
        w.r.t the weight @self.w is equal to feature vector @self.features

        Params
        ------
        state: int
            current state
        n_states: int
            number of states

        Return
        ------
        grad: np.ndarray
            gradient w.r.t @self.w at the current state
        '''
        state /= float(random_walk.n_states)
        feature_vector = self.get_feature_vector(state)
        grad = feature_vector
        return grad


    def get_value(self, state, random_walk):
        '''
        Get value function at state @state
        value function is equal to dot product of its feature vector and weight
        

        Params
        ------
        state: int
            current state
        n_states: int
            number of states

        Return
        ------
        value_func: float
            value function at state @state
        '''
        state /= float(random_walk.n_states)
        feature_vector = self.get_feature_vector(state)
        value_func = np.dot(self.w, feature_vector)
        return value_func


def gradient_mc(value_func, random_walk, alpha, mu=None):
    '''
    Gradient Monte Carlo

    Params
    ------
    value_func
    random_walk: RandomWalk
    alpha: float
        step size
    mu: np.ndarray
        state distribution
    '''
    state = random_walk.start_state
    trajectory = [state]

    while not random_walk.is_terminal(state):
        action = random_policy(random_walk)
        next_state = random_walk.get_next_state(state, action)
        trajectory.append(next_state)
        state = next_state
    reward = random_walk.get_reward(state)

    # since reward at every states except terminal ones is 0, and discount factor gamma = 1,
    # the return at each state is equal to the reward at the terminal state.
    for state in trajectory[:-1]:
        value_func.w += alpha * (reward - value_func.get_value(state, random_walk)) \
            * value_func.get_grad(state, random_walk)
        if mu is not None:
            mu[state] += 1


def gradient_mc_state_aggregation_plot(random_walk, true_value):
    '''
    Plot gradient MC w/ state aggregation

    Params
    ------
    random_walk: RandomWalk
    true_value: np.ndarray
        true values
    '''
    alpha = 2e-5
    n_groups = 10
    n_eps = 100000
    mu = np.zeros(n_states + 2)
    state_agg = StateAggregationValueFunction(n_groups, random_walk.n_states)

    for _ in trange(n_eps):
        gradient_mc(state_agg, random_walk, alpha, mu)

    mu /= np.sum(mu)
    value_funcs = [state_agg.get_value(state, random_walk)
        for state in random_walk.states]

    fig, ax1 = plt.subplots()
    value_func_plot = ax1.plot(random_walk.states, value_funcs, 
        label=r'Approximate MC value $\hat{v}$', color='blue')
    true_value_plot = ax1.plot(random_walk.states, true_value[1: -1], 
        label=r'True value $v_\pi$', color='red')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value scale')
    ax2 = ax1.twinx()
    state_dist_plot = ax2.plot(random_walk.states, mu[1: -1], 
        label=r'State distribution $\mu$', color='gray')
    ax2.set_ylabel('Distribution scale')
    plots = value_func_plot + true_value_plot + state_dist_plot
    labels = [l.get_label() for l in plots]
    plt.legend(plots, labels, loc=0)
    plt.savefig('./gradient_mc_state_agg.png')
    plt.close()


def n_step_semi_gradient_td(value_func, random_walk, n, alpha, gamma):
    '''
    n-step semi-gradient TD

    Params
    ------
    value_func
    random_walk: RandomWalk
    n: int
        number of step
    alpha: float
        step size
    gamma: float
        discount factor
    '''
    state = random_walk.start_state
    states = [state]

    T = float('inf')
    t = 0
    rewards = [0] # dummy reward to save the next reward as R_{t+1}

    while True:
        if t < T:
            action = random_policy(random_walk)
            next_state, reward = random_walk.take_action(state, action)
            states.append(next_state)
            rewards.append(reward)
            if random_walk.is_terminal(next_state):
                T = t + 1
        tau = t - n + 1
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += np.power(gamma, i - tau - 1) * rewards[i]
            if tau + n < T:
                G += np.power(gamma, n) * value_func.get_value(states[tau + n], random_walk)
            if not random_walk.is_terminal(states[tau]):
                value_func.w += alpha * (G - value_func.get_value(states[tau], random_walk)) * \
                    value_func.get_grad(states[tau])
        t += 1
        if tau == T - 1:
            break
        state = next_state


def semi_gradient_td_0_plot(random_walk, true_value):
    '''
    Plot semi-gradient TD(0)

    Params
    ------
    random_walk: RandomWalk
    true_value: np.ndarray
        true values
    '''
    alpha = 2e-4
    n_groups = 10
    n_eps = 100000
    gamma = 1
    state_agg = StateAggregationValueFunction(n_groups, random_walk.n_states)

    for _ in trange(n_eps):
        n_step_semi_gradient_td(state_agg, random_walk, 1, alpha, gamma)

    value_funcs = [state_agg.get_value(state, random_walk)
        for state in random_walk.states]

    plt.plot(random_walk.states, value_funcs, 
        label=r'Approximate TD value $\hat{v}$', color='blue')
    plt.plot(random_walk.states, true_value[1: -1], 
        label=r'True value $v_\pi$', color='red')
    plt.xlabel('State')
    plt.ylabel('Value scale')
    plt.legend()


def n_step_semi_gradient_td_plot(random_walk, true_value):
    '''
    Plot n-step semi-gradient TD

    Params
    ------
    random_walk: RandomWalk
    true_value: np.ndarray
        true values
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
                state_agg = StateAggregationValueFunction(n_groups, random_walk.n_states)
                for _ in trange(n_eps):
                    n_step_semi_gradient_td(state_agg, random_walk, n, alpha, gamma)
                    state_values = np.array([state_agg.get_value(state, random_walk) 
                        for state in random_walk.states])
                    rmse = np.sqrt(np.sum(np.power(state_values - true_value[1: -1], 2) 
                        / random_walk.n_states))
                    errors[n_i, alpha_i] += rmse

    errors /= n_eps * n_runs

    for i in range(0, len(ns)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (ns[i]))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()


def semi_gradient_td_plot(random_walk, true_value):
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    semi_gradient_td_0_plot(random_walk, true_value)
    plt.subplot(122)
    n_step_semi_gradient_td_plot(random_walk, true_value)
    plt.savefig('./semi_gradient_td.png')
    plt.close()


def gradient_mc_bases_plot(random_walk, true_value):
    '''
    Plot gradient Monte Carlo w/ Fourier and polynomial bases

    Params
    ------
    random_walk: RandomWalk
    true_value: np.ndarray
        true values
    '''
    orders = [5, 10, 20]
    n_runs = 1
    n_eps = 5000

    bases = [
        {'method': 'Polynomial', 'alpha': 1e-4},
        {'method': 'Fourier', 'alpha': 5e-5}
    ]

    errors = np.zeros((len(bases), len(orders), n_eps))
    for i_basis, basis in enumerate(bases):
        for i_order, order in enumerate(orders):
            print(f'{basis["method"]} basis, order={order}')
            for _ in range(n_runs):
                value_func = BasesValueFunction(order, basis['method'])
                for ep in trange(n_eps):
                    gradient_mc(value_func, random_walk, basis['alpha'])
                    state_values = np.array([value_func.get_value(state, random_walk) 
                        for state in random_walk.states])
                    rmse = np.sqrt(np.mean(np.power(state_values - true_value[1: -1], 2)))
                    errors[i_basis, i_order, ep] += rmse

    errors /= n_runs
    for i_basis, basis in enumerate(bases):
        for i_order, order in enumerate(orders):
            plt.plot(errors[i_basis, i_order, :], label='%s basis, order = %d' % (basis['method'], order))
    plt.xlabel('Episodes')
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('./gradient_mc_bases.png')
    plt.close()


if __name__ == '__main__':
    n_states = 1000
    start_state = 500
    transition_radius = 100
    random_walk = RandomWalk(n_states, start_state, transition_radius)
    true_value = get_true_value(random_walk)

    # gradient_mc_state_aggregation_plot(random_walk, true_value)
    # semi_gradient_td_plot(random_walk, true_value)
    gradient_mc_bases_plot(random_walk, true_value)

