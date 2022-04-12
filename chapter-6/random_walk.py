import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import string


class RandomWalk:


    def __init__(self, n_states, start_state):
        self.n_states = n_states
        self.start_state = start_state
        self.states = np.arange(1, n_states + 1)
        self.state_labels = list(string.ascii_uppercase)[:n_states]
        self.end_states = [0, n_states + 1]
        self.actions = [-1, 1]
        self.rewards = [0, 1]


    def is_terminal(self, state):
        '''
        Whether state @state is an end state

        Params
        ------
        state: int
            current state
        '''
        return state in self.end_states


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
        if state == self.end_states[1]:
            reward = self.rewards[1]
        else:
            reward = self.rewards[0]

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


def get_true_value(n_states):
    '''
    Compute true values
    '''
    true_value = [1.0 * x / (n_states + 1) for x in range(1, n_states + 1)]
    return true_value


def random_policy(random_walk):
    '''
    Choose an action randomly

    Params
    ------
    random_walk: RandomWalk
    '''
    return np.random.choice(random_walk.actions)


def temporal_difference(V, random_walk, alpha, gamma, batch_update=False):
    '''
    TD(0) w/ & without batch updating

    Params
    ------
    V: np.ndarray
        value function
    random_walk: RandomWalk
    alpha: float
        step size
    gamma: float
        discount factor
    batch_update: boolean
        is batch updating
    '''
    state = random_walk.start_state
    trajectory = [[state, 0]]

    while True:
        action = random_policy(random_walk)
        next_state, reward = random_walk.take_action(state, action)
        trajectory.append([next_state, reward])
        if not batch_update:
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state
        if random_walk.is_terminal(state):
            break

    return trajectory


def monte_carlo(V, random_walk, alpha, gamma, batch_update=False):
    '''
    Constant-alpha Monte Carlo

    Params
    ------
    V: np.ndarray
        value function
    random_walk: RandomWalk
    alpha: float
        step size
    gamma: float
        discount factor
    batch_update: boolean
        is batch updating
    '''
    state = random_walk.start_state
    trajectory = [[state, 0]]

    while True:
        action = random_policy(random_walk)
        next_state, reward = random_walk.take_action(state, action)
        trajectory.append([next_state, reward])
        state = next_state

        if random_walk.is_terminal(state):
            break

    # the return at each state is equal to the reward at the terminal state.
    if not batch_update:
        for state_, _ in trajectory[:-1]:
            V[state_] += alpha * (reward - V[state_])

    return trajectory


def get_state_values(random_walk, true_value, n_eps, alpha, gamma):
    V = np.full(random_walk.n_states + 2, 0.5)
    V[0] = V[-1] = 0
    eps = [0, 1, 10, 100]

    plt.plot(random_walk.state_labels, true_value, label='true values')

    for ep in range(n_eps + 1):
        if ep in eps:
            plt.plot(random_walk.state_labels, V[1:-1], label=str(ep) + ' episodes')
        _ = temporal_difference(V, random_walk, alpha, gamma)

    plt.xlabel('State')
    plt.ylabel('Estimated value')
    plt.legend()


def get_rmse(random_walk, true_value, n_eps, gamma):
    V = np.full(random_walk.n_states + 2, 0.5)
    V[0] = V[-1] = 0
    methods = [
        {
            'name': 'TD',
            'alphas': [0.05, 0.1, 0.15],
            'func': temporal_difference,
            'linestyle': 'solid'
        },
        {
            'name': 'MC',
            'alphas': [0.01, 0.02, 0.03, 0.04],
            'func': monte_carlo,
            'linestyle': 'dashdot'
        }
    ]
    n_runs = 100

    for method in methods:
        for alpha in method['alphas']:
            print(f'{method["name"]} method, alpha={alpha}', end='')
            total_errors = np.zeros(n_eps)
            for _ in tqdm(range(n_runs)):
                V_ = V.copy()
                errors = []
                for _ in range(n_eps):
                    rmse = np.sqrt(np.sum(np.power(V_[1:-1] - true_value, 2) / random_walk.n_states))
                    errors.append(rmse)
                    _ = method['func'](V_, random_walk, alpha, gamma)
                total_errors += np.asarray(errors)
            total_errors /= n_runs
            plt.plot(total_errors, label=method['name'] + ', alpha = %.02f' % (alpha), 
                linestyle=method['linestyle'])
            print()
    plt.xlabel('Episodes')
    plt.ylabel('RMS')
    plt.legend()


def rmse_batch_updating(random_walk, true_value, n_eps, alpha, gamma):
    V = np.full(random_walk.n_states + 2, -1.0)
    V[0] = 0
    V[-1] = 1
    methods = [
        {
            'name': 'TD',
            'func': temporal_difference,
        },
        {
            'name': 'MC',
            'func': monte_carlo,
        }
    ]
    n_runs = 100

    for method in methods:
        print(f'{method["name"]} method', end='')
        total_errors = np.zeros(n_eps)
        for _ in tqdm(range(n_runs)):
            V_ = V.copy()
            errors = []
            trajectories = []
            for _ in range(n_eps):
                trajectory = method['func'](V_, random_walk, alpha, gamma, True)
                trajectories.append(trajectory)

                while True:
                    # old_values = V_.copy()
                    delta = np.zeros(random_walk.n_states + 2)

                    for trajectory_ in trajectories:
                        for i in range(len(trajectory_) - 1):
                            state = trajectory_[i][0]
                            next_state = trajectory_[i+1][0]
                            if method['name'] == 'TD':
                                reward = trajectory_[i][1]
                                # V_[state] += alpha * (reward + gamma * V_[next_state] - V_[state])
                                delta[state] += alpha * (reward + gamma * V_[next_state] - V_[state])
                            else:
                                return_ = trajectory_[-1][1]
                                # V_[state] += alpha * (return_ - V_[state])
                                delta[state] += alpha * (return_ - V_[state])
                    # if np.abs(np.sum(V_ - old_values)) < 1e-3:
                    if np.sum(np.abs(delta)) < 1e-3:
                        break
                    V_ += delta

                rmse = np.sqrt(np.sum(np.power(V_[1:-1] - true_value, 2)) / random_walk.n_states)
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
    random_walk = RandomWalk(n_states, start_state)
    n_eps = 100
    alpha = 0.1
    gamma = 1
    true_value = get_true_value(n_states)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    get_state_values(random_walk, true_value, n_eps, alpha, gamma)

    plt.subplot(1, 2, 2)
    get_rmse(random_walk, true_value, n_eps, gamma)
    plt.tight_layout()
    plt.savefig('./random_walk.png')
    plt.close()

    print('Batch updating')
    batch_alpha = 0.001
    rmse_batch_updating(random_walk, true_value, n_eps, batch_alpha, gamma)

