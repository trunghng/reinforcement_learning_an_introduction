import numpy as np
import gym
import matplotlib.pyplot as plt


def epsilon_greedy(epsilon, Q, n_actions, state):
    '''
    Choose action according to epsilon-greedy policy

    Params:
    -------
    epsilon: float
    Q: np.ndarray
        action-value function
    n_actions: int
        number of actions
    state: [int, int]
        current state

    Return
    ------
    action: (int, int)
    '''
    if not np.random.binomial(1, epsilon):
        action = np.argmax(Q[state[0], state[1]])
    else:
        action = np.random.randint(n_actions)
    return action


def discretize_state(env, position, velocity):
    '''
    Discretize state
        [position, velociy] => [int(position * 10), int(velocity * 100)]

    Params
    ------
    env: OpenAI MoutainCar env
    position: float
        position of the car
    velocity: float
        velocity of the car
    '''
    position_low, velocity_low = env.low
    discretized_position = int((position - position_low) * 10)
    discretized_velocity = int((velocity - velocity_low) * 100)
    return discretized_position, discretized_velocity


def q_learning(Q, env, epsilon, alpha, gamma, current_ep):
    '''
    Q-learning algorithm

    Params
    ------
    Q: np.ndarray
        action-value function
    env: OpenAI MoutainCar env
    epsilon: float
        epsilon greedy param
    alpha: float
        step size
    gamma: float
        discount factor
    current_ep: int
        current episode

    Return
    ------
    total_reward: int
        total reward of the episode
    '''
    start_state = env.reset()
    state = discretize_state(env, start_state[0], start_state[1])
    n_actions = env.action_space.n
    total_reward = 0

    while True:
        if current_ep + 20 >= n_eps:
            env.render()

        action = epsilon_greedy(epsilon, Q, n_actions, state)
        next_state_, reward, terminated, _ = env.step(action)
        next_state = discretize_state(env, next_state_[0], next_state_[1])

        if terminated and next_state_[0] >= 0.5:
            Q[state[0], state[1], action] = reward
        else:
            delta = alpha * (reward + gamma 
                * np.max(Q[next_state[0], next_state[1]]) 
                - Q[state[0], state[1], action])
            Q[state[0], state[1], action] += delta

        total_reward += reward
        state = next_state

        if terminated:
            break

    return total_reward


if __name__ == '__main__':
    # state = [position, velocity]
    # position_range = [-1.2, 0.6]
    # velocity_range = [-0.7, 0.7]
    env = gym.make('MountainCar-v0')
    env.reset()
    position_low, velocity_low = env.low
    position_high, velocity_high = env.high
    max_position, max_velocity = discretize_state(env, position_high, velocity_high)
    min_position, min_velocity = discretize_state(env, position_low, velocity_low)
    n_positions = max_position - min_position + 1
    n_velocities = max_velocity - min_velocity + 1
    n_actions = env.action_space.n

    Q = np.zeros((n_positions, n_velocities, n_actions))
    n_eps = 5000
    min_epsilon = 0
    epsilon = 0.1
    alpha = 0.5
    gamma = 0.9
    min_alpha = 0.01
    reward_list = []
    ave_reward_list = []
    reduction = (alpha - min_alpha) / n_eps

    for ep in range(n_eps):
        total_reward = q_learning(Q, env, epsilon, alpha, gamma, ep)

        if alpha > min_alpha:
            alpha -= reduction

        reward_list.append(total_reward)

        if (ep + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (ep + 1) % 100 == 0:    
            print('Episode {} Average Reward: {}'.format(ep + 1, ave_reward))

    env.close()
    plt.plot(100 * (np.arange(len(ave_reward_list)) + 1), ave_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('./mountain_car.png')
    plt.close()  
    