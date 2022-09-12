import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class Task:

    def __init__(self, n_states, actions, branching_factor, terminate_prob):
        """
        Parameters:
        -----------
        n_states: int
            number of states, count from 0 to n_states-1; terminal state is denoted as state n_states
        actions: list
            list of actions
        branching_factor: int
            branching factor
        terminate_prob: float
            probability of termination in each transition
        """
        self._n_states = n_states
        self._actions = actions
        self._branching_factor = branching_factor
        self._terminate_prob = terminate_prob
        self._transition_matrix = np.random.randint(n_states, size=(n_states, len(actions), branching_factor))
        self._reward_function = np.random.randn(n_states, len(actions), branching_factor)


    @property
    def n_states(self):
        return self._n_states


    @property
    def actions(self):
        return self._actions


    @property
    def branching_factor(self):
        return self._branching_factor


    @property
    def terminate_prob(self):
        return self._terminate_prob


    @property
    def transition_matrix(self):
        return self._transition_matrix


    @property
    def reward_function(self):
        return self._reward_function


    def take_action(self, state, action):
        if np.random.binomial(1, self.terminate_prob):
            next_state = self.n_states
            reward = 0
        else:
            next_state_idx = np.random.randint(self.branching_factor)
            next_state = self.transition_matrix[state, action, next_state_idx]
            reward = self.reward_function[state, action, next_state_idx]
        return next_state, reward


    def is_terminal(self, state):
        return state == self.n_states


def epsilon_greedy(epsilon, Q, state, task):
    if np.random.binomial(1, epsilon):
        action = np.random.choice(task.actions)
    else:
        values = Q[state, :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values) 
            if value_ == np.max(values)])
    return action


def value_eval(Q, task):
    """
    evaluate value of the start state using MC method
    """
    runs = 1000
    returns = []

    for _ in range(runs):
        state = 0
        rewards = 0
        while not task.is_terminal(state):
            values = Q[state, :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values) 
                if value_ == np.max(values)])
            next_state, reward = task.take_action(state, action)
            rewards += reward
            state = next_state
        returns.append(rewards)
    return np.mean(returns)


def on_policy(task, max_iters, gamma, eval_interval):
    efficiency = []
    state = 0
    Q = np.zeros((task.n_states, len(task.actions)))

    for step in tqdm(range(max_iters)):
        if task.is_terminal(state):
            next_state = 0
        else:
            action = epsilon_greedy(epsilon, Q, state, task)
            next_state, reward = task.take_action(state, action)

            next_states = task.transition_matrix[state, action]
            rewards = task.reward_function[state, action]

            non_terminate_prob = 1 - task.terminate_prob
            Q[state, action] = non_terminate_prob * np.mean(rewards + gamma 
                * np.max(Q[next_states, :], axis=1))

        state = next_state
        if step % eval_interval == 0:
            value = value_eval(Q, task)
            efficiency.append([step, value])

    return efficiency


def uniform(task, max_iters, gamma, eval_interval):
    efficiency = []
    Q = np.zeros((task.n_states, len(task.actions)))

    for step in tqdm(range(max_iters)):
        state = np.random.randint(task.n_states)
        action = np.random.choice(task.actions)

        next_states = task.transition_matrix[state, action]
        rewards = task.reward_function[state, action]

        non_terminate_prob = 1 - task.terminate_prob
        Q[state, action] = non_terminate_prob * np.mean(rewards + gamma 
            * np.max(Q[next_states, :], axis=1))

        if step % eval_interval == 0:
            value = value_eval(Q, task)
            efficiency.append([step, value])

    return efficiency


if __name__ == '__main__':
    n_states_list = [1000, 10000]
    actions = [0, 1]
    branching_factors = [1, 3, 10]
    terminate_prob = 0.1
    epsilon = 0.1
    max_iters = 20000
    gamma = 1
    eval_interval = 200
    methods = {'on-policy': on_policy, 'uniform': uniform}
    n_tasks = 20

    fig, axs = plt.subplots(len(n_states_list), 1, squeeze=True, figsize=(10, 20))
    axs = np.array(axs).reshape(-1)

    for ax, n_states in zip(axs, n_states_list):
        ax.set_title(f'{n_states} states')
        ax.set_xlabel('Computation time, in expected updates')
        ax.set_ylabel('Value of start state under greedy policy')

        for branching_factor in branching_factors:
            tasks = [Task(n_states, actions, branching_factor, terminate_prob) 
                for _ in range(n_tasks)]
            for method_name in methods:
                steps = None
                values = []
                for i, task in enumerate(tasks):
                    print(f'{method_name}, n_states={n_states}, b={branching_factor}, task {i}')
                    steps, values_ = zip(*methods[method_name](task, max_iters, 
                        gamma, eval_interval))
                    time.sleep(0.1)
                    values.append(values_)
                values = np.mean(np.asarray(values), axis=0)
                ax.plot(steps, values, label=f'{method_name}, b={branching_factor}')
        ax.legend()
    fig.tight_layout()
    plt.savefig('./trajectory_sampling.png')
    plt.close()
