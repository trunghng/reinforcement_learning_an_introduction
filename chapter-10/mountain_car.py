import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import trange
import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from tile_coding import IHT, tiles
import math


class ValueFunction:

    def __init__(self, n_tilings, env):
        '''
        n_tilings: int
            number of tilings used for tile coding
        env: OpenAI's MountainCar env
        '''
        self.n_tilings = n_tilings
        self.position_scale = n_tilings / (env.high[0] - env.low[0])
        self.velocity_scale = n_tilings / (env.high[1] - env.low[1])
        # size = math.ceil((self.position_scale + 1) * (self.velocity_scale + 1) * n_tilings)
        size = 4096
        self.iht = IHT(size)
        self.w = np.zeros(size)
        self.env = env


    def get_active_tiles(self, position, velocity, action):
        '''
        Get (indices of) active tiles corresponding to 
        state-action pair [[@position, @velocity], @action]
        (i.e., index of the tile in each tilings where the value = 1)

        Params
        ------
        position: float
            current position of the car
        velocity: float
            current velocity of the car
        action: int
            action taken at the current state

        Return
        ------
        active_tiles: list
        '''
        active_tiles = tiles(self.iht, self.n_tilings, [position * 
            self.position_scale, self.velocity_scale * velocity], [action])
        return active_tiles


    def get_value(self, position, velocity, action):
        '''
        Get action-value of state-action pair [[@position, @velocity], @action]
        Since the feature vector is one-hot and 
        we are using linear function approx
        => value at [[@position, @velocity], @action] is exactly the 
            total of weight corresponding to [[@position, @velocity], @action]

        Params
        ------
        position: float
            current position of the car
        velocity: float
            current velocity of the car
        action: int
            action taken at the current state
        '''
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.w[active_tiles])


    def learn(self, position, velocity, action, target, alpha):
        '''
        Update weight vector

        Params
        ------
        position: float
            current position of the car
        velocity: float
            current velocity of the car
        action: int
            action taken at the current state
        alpha: float
            step size param
        '''
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimate = np.sum(self.w[active_tiles])
        error = target - estimate
        for tile in active_tiles: 
            self.w[tile] += alpha / self.n_tilings * error


    def cost_to_go(self, position, velocity, n_actions):
        '''
        Get cost-to-go at the current state [@position, @velocity]

        Params
        ------
        position: float
            current position of the car
        velocity: float
            current velocity of the car
        n_actions: int
            number of actions
        '''
        costs = np.array([self.get_value(position, velocity, action_) 
            for action_ in range(n_actions)])
        return -np.max(costs)


def epsilon_greedy(epsilon, value_function, position, velocity, n_actions):
    '''
    Epsilon-greedy policy

    Params:
    -------
    epsilon: float
    value_function: ValueFunction
        action-value function
    position: float
        current position of the car
    velocity: float
        current velocity of the car
    n_actions: int
        number of actions

    Return
    ------
    action: int
    '''
    if not np.random.binomial(1, epsilon):
        values = np.array([value_function.get_value(position, velocity, action_) 
            for action_ in range(n_actions)])
        action = np.argmax(values)
    else:
        action = np.random.randint(n_actions)
    return action


def episodic_semi_gradient_sarsa(value_function, env, alpha, gamma, epsilon, 
                                    current_ep, n_eps):
    '''
    Episodic Semi-gradient Sarsa algorithm

    Params
    ------
    value_function: ValueFunction
        action-value function
    env: OpenAI's MountainCar
    alpha: float
        step size
    gamma: float
        discount factor
    epsilon: float
        epsilon greedy param
    current_ep: int
        current epside
    n_eps: int
        total number of episodes
    '''
    n_actions = env.action_space.n
    state = env.reset()
    action = epsilon_greedy(epsilon, value_function, 
        state[0], state[1], n_actions)

    while True:
        if current_ep + 10 >= n_eps:
            env.render()

        next_state, reward, terminated, _ = env.step(action)
        position, velocity = state
        next_position, next_velocity = next_state
        if terminated:
            value_function.learn(position, velocity, action, reward, alpha)
            break
        next_action = epsilon_greedy(epsilon, 
            value_function, next_position, next_velocity, n_actions)
        value_function.learn(position, velocity, action, 
            reward + gamma * value_function.get_value(
            next_position, next_velocity, next_action), alpha)
        state = next_state
        action = next_action


def episodic_semi_gradient_n_step_sarsa(value_function, env, n,
                                    alpha, gamma, epsilon):
    '''
    Episodic Semi-gradient n-step Sarsa algorithm

    Params
    ------
    value_function: ValueFunction
        action-value function
    env: OpenAI's MountainCar
    n: int
        n-step
    alpha: float
        step size
    gamma: float
        discount factor
    epsilon: float
        epsilon greedy param
    '''
    n_actions = env.action_space.n
    state = env.reset()
    action = epsilon_greedy(epsilon, value_function, 
        state[0], state[1], n_actions)
    states = [state]
    actions = [action]
    rewards = [0] # 0 is a dummy reward
    T = float('inf')
    t = 0

    while True:
        if t < T:
            next_state, reward, terminated, _ = env.step(actions[t])
            states.append(next_state)
            rewards.append(reward)

            if terminated:
                T = t + 1
            else:
                next_action = epsilon_greedy(epsilon, value_function,
                    next_state[0], next_state[1], n_actions)
                actions.append(next_action)

        tau = t - n + 1
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += np.power(gamma, i - tau - 1) * rewards[i]
            if tau + n < T:
                G += np.power(gamma, n) * value_function.get_value(
                    states[tau + n][0], states[tau + n][1], actions[tau + n])
                value_function.learn(states[tau][0], states[tau][1], 
                    actions[tau], G, alpha)

        t += 1
        if tau == T - 1:
            break

    return t


def episodic_semi_gradient_sarsa_plot():
    n_eps = 9000
    alpha = 0.3
    gamma = 1
    epsilon = 0.1
    n_tilings = 8
    plot_eps = [0, 99, 399, 999, 3999, n_eps - 1]
    fig = plt.figure(figsize=(24, 16))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    plot_count = 0
    env = gym.make('MountainCar-v0')
    env.reset()
    value_function = ValueFunction(n_tilings, env)

    for ep in trange(n_eps):
        episodic_semi_gradient_sarsa(value_function, env, alpha, 
            gamma, epsilon, ep, n_eps)
        if ep in plot_eps:
            ax = fig.add_subplot(2, 3, plot_count + 1, projection='3d')
            plot_count += 1
            positions = np.linspace(env.low[0], env.high[0])
            velocities = np.linspace(env.low[1], env.high[1])
            axis_x = []
            axis_y = []
            axis_z = []
            for position in positions:
                for velocity in velocities:
                    axis_x.append(position)
                    axis_y.append(velocity)
                    axis_z.append(value_function.cost_to_go(position, 
                        velocity, env.action_space.n))

            ax.scatter(axis_x, axis_y, axis_z)
            ax.set_xlabel('Position')
            ax.set_ylabel('Velocity')
            ax.set_zlabel('Cost to go')
            ax.set_title('Episode %d' % (ep + 1))

    plt.savefig('./mountain-car-ep-semi-grad-sarsa.png')
    plt.close()


def episodic_semi_gradient_n_step_sarsa_plot():
    runs = 100
    n_eps = 500
    n_tilings = 8
    ns = [1, 8]
    alphas = [0.5, 0.3]
    gamma = 1
    epsilon = 0
    env = gym.make('MountainCar-v0')
    env.reset()

    steps = np.zeros((len(alphas), n_eps))
    for run in range(runs):
        value_functions = [ValueFunction(n_tilings, env) for _ in alphas]
        for alpha_idx in range(len(alphas)):
            for ep in trange(n_eps):
                step = episodic_semi_gradient_n_step_sarsa(value_functions[alpha_idx], 
                    env, ns[alpha_idx], alphas[alpha_idx], gamma, epsilon)
                steps[alpha_idx, ep] += step

    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='n = %d' % (ns[i]))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    # plt.ylim([100, 1000])
    plt.yscale('log')
    plt.legend()

    plt.savefig('./mountain-car-ep-semi-grad-n-step-sarsa.png')
    plt.close()


if __name__ == '__main__':
    # episodic_semi_gradient_sarsa_plot()
    episodic_semi_gradient_n_step_sarsa_plot()
