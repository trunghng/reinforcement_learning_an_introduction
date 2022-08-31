import sys
import gym
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from tile_coding import tiles, IHT
from tqdm import trange


class SarsaLambda:

    def __init__(self, env, lambda_, alpha, gamma, epsilon, 
            n_tilings, is_accumulated, max_steps):
        '''
        Params
        ------
        env: OpenAI's MountainCar
        lambda_: float
            trace decay param
        gamma: float
            discount factor
        epsilon: float
            epsilon greedy param
        n_tilings: int
            number of tilings used for tile coding
        is_accumulated: boolean
        max_steps: int
            maximum number of steps
        '''
        self.env = env
        self.position_scale = n_tilings / (env.high[0] - env.low[0])
        self.velocity_scale = n_tilings / (env.high[1] - env.low[1])
        self.lambda_ = lambda_
        self.alpha = alpha / n_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.is_accumulated = is_accumulated
        self.max_steps = max_steps
        self.iht = IHT(2048)
        self.weights = np.zeros(2048)
        self.trace = np.zeros(2048)


    def reset(self):
        '''
        Episode initialization
        '''
        self.start_state = self.env.reset()


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
        active_tiles = tiles(self.iht, self.n_tilings, 
            [position * self.position_scale, velocity 
            * self.velocity_scale], [action])
        return active_tiles


    def get_value(self, position, velocity, action, terminated=None):
        '''
        Get state-action value of state-action pair [[@position, @velocity], @action]
        Since the feature vector is one-hot and we are using linear function approx
        => value at [[@position, @velocity], @action] is exactly the 
            total of weight corresponding to [[@position, @velocity], @action]

        Params
        ------
        position: float
            position of the car
        velocity: float
            velocity of the car
        action: int
            action taken at the state [@position, @velocity]
        terminated: is terminated

        Return
        ------
        value: float
            state-action value
        '''
        if terminated is not None and terminated:
            return 0
        active_tiles = self.get_active_tiles(position, velocity, action)
        value = np.sum(self.weights[active_tiles])
        return value


    def epsilon_greedy(self, epsilon, position, velocity):
        '''
        Epsilon greedy policy

        Params
        ------
        epsilon: float
            epsilon greedy param
        position: float
            current position of the car
        velocity: float
            current velocity of the car

        Return
        ------
        action: int
            chosen action
        '''
        n_actions = self.env.action_space.n
        if not np.random.binomial(1, epsilon):
            values = np.array([self.get_value(position, velocity,
                action_) for action_ in range(n_actions)])
            action = np.argmax(values)
        else:
            action = np.random.randint(n_actions)
        return action


    def update_accumulating_trace(self, active_tiles):
        '''
        Update trace as an accumulating trace
        z_{t+1} := gamma * lambda * z_{t} + gradient{hat{q}(S_t,A_t,w_t)}

        Params
        ------
        active_tiles: list<int>
            list of (indices of) active tiles
        '''
        self.trace *= self.gamma * self.lambda_
        self.trace[active_tiles] += 1


    def update_replacing_trace(self, active_tiles):
        '''
        Update trace as a replacing trace
        z_{i,t} := 1                          if x_{i,t} = 1
                := gamma * lambda * z_{i,t-1} if x_{i,t} = 0

        Params
        ------
        active_tiles: list<int>
            list of (indices of) active tiles
        '''
        boolean_active_tiles = np.in1d(np.arange(self.trace.shape[0]), active_tiles)
        self.trace[boolean_active_tiles] = 1
        self.trace[~boolean_active_tiles] *= self.gamma * self.lambda_


    def learn(self, position, velocity, action, target):
        '''
        Update weight vector @self.weights

        Params
        ------
        position: float
            position of the car
        velocity: float
            velocity of the car
        action: int
            action taken at state [@position, @velocity]
        target: float
            target value
        '''
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimate = np.sum(self.weights[active_tiles])
        td_error = target - estimate
        if self.is_accumulated:
            self.update_accumulating_trace(active_tiles)
        else:
            self.update_replacing_trace(active_tiles)
        self.weights += self.alpha * td_error * self.trace


    def run(self):
        '''
        Perform an episode

        Return
        ------
        steps: int
            number of steps in the episode
        '''
        self.reset()
        steps = 0
        position, velocity = self.start_state
        action = self.epsilon_greedy(self.epsilon, position, velocity)

        while True:
            next_state, reward, terminated, _ = self.env.step(action)
            next_position, next_velocity = next_state
            next_action = self.epsilon_greedy(
                self.epsilon, next_position, next_velocity)
            steps += 1
            target = reward + self.gamma * self.get_value(next_position, 
                next_velocity, next_action, terminated)
            self.learn(position, velocity, action, target)
            position = next_position
            velocity = next_velocity
            action = next_action

            if terminated:
                break
            elif steps >= self.max_steps:
                print('Max #steps reached!')
                break

        return steps


class TrueOnlineSarsaLambda:


    def __init__(self, env, lambda_, alpha, gamma, epsilon, n_tilings, n_eps, max_steps):
        '''
        Params
        ------
        env: OpenAI's MountainCar
        lambda_: float
            trace decay param
        gamma: float
            discount factor
        epsilon: float
            epsilon greedy param
        n_tilings: int
            number of tilings used for tile coding
        '''
        self.env = env
        self.position_scale = n_tilings / (env.high[0] - env.low[0])
        self.velocity_scale = n_tilings / (env.high[1] - env.low[1])
        self.lambda_ = lambda_
        self.alpha = alpha / n_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.n_eps = n_eps
        self.max_steps = max_steps
        self.weights = np.zeros(2048)
        self.trace = np.zeros(2048)
        self.iht = IHT(2048)


    def reset(self):
        '''
        Episode initialization
        '''
        self.start_state = self.env.reset()


    def get_active_tiles(self, position, velocity, action, terminated=None):
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
        terminated: boolean
            is terminated

        Return
        ------
        active_tiles: list
        '''
        if terminated is not None and terminated:
            return []
        active_tiles = tiles(self.iht, self.n_tilings, 
            [position * self.position_scale, velocity 
            * self.velocity_scale], [action])
        return active_tiles


    def get_value(self, position, velocity, action, terminated=None):
        if terminated is not None and terminated:
            return 0
        active_tiles = self.get_active_tiles(position, velocity, action)
        value = np.sum(self.weights[active_tiles])
        return value


    def get_value_with_active_tiles(self, active_tiles, terminated=None):
        if terminated is not None and terminated:
            return 0
        value = np.sum(self.weights[active_tiles])
        return value


    def epsilon_greedy(self, epsilon, position, velocity):
        '''
        Epsilon greedy policy

        Params
        ------
        epsilon: float
            epsilon greedy param
        position: float
            current position of the car
        velocity: float
            current velocity of the car

        Return
        ------
        action: int
            chosen action
        '''
        n_actions = self.env.action_space.n
        if not np.random.binomial(1, epsilon):
            values = np.array([self.get_value(position, velocity,
                action_) for action_ in range(n_actions)])
            action = np.argmax(values)
        else:
            action = np.random.randint(n_actions)
        return action


    def learn(self, error):
        '''
        Update weight vector @self.weights

        Params
        ------
        error: float
        '''
        self.weights += self.alpha * error


    def update_trace(self, active_tiles):
        '''
        Update trace vector as a dutch trace

        Params
        ------
        active_tiles: list<int>
            list of (indices of) active tiles
        '''
        delta = 1 - self.alpha * self.gamma * self.lambda_ \
            * np.sum(self.trace[active_tiles])
        self.trace *= self.gamma * self.lambda_
        self.trace[active_tiles] += delta


    def run(self, current_ep):
        '''
        Perform an episode

        Return
        ------
        total_reward: int
            total reward in the episode
        '''
        self.reset()
        position, velocity = self.start_state
        action = self.epsilon_greedy(self.epsilon, position, velocity)
        active_tiles = self.get_active_tiles(position, velocity, action)
        old_action_value = 0
        total_reward = 0
        step = 0

        while True:
            step += 1
            if current_ep + 20 >= self.n_eps:
                self.env.render()

            next_state, reward, terminated, _ = self.env.step(action)
            total_reward += reward
            next_position, next_velocity = next_state
            next_action = self.epsilon_greedy(self.epsilon, 
                next_position, next_velocity)
            next_active_tiles = self.get_active_tiles(next_position, 
                next_velocity, next_action, terminated)
            action_value = self.get_value_with_active_tiles(active_tiles)
            next_action_value = self.get_value_with_active_tiles(next_active_tiles, terminated)
            td_error = reward + self.gamma * next_action_value - action_value
            self.update_trace(active_tiles)
            error = (td_error + action_value - old_action_value) * self.trace
            error[active_tiles] -= action_value - old_action_value
            self.learn(error)
            old_action_value = next_action_value
            active_tiles = next_active_tiles
            action = next_action

            if terminated:
                break
            elif step >= self.max_steps:
                print('Max #steps reached!')
                break

        return total_reward


def sarsa_lambda_plot():
    env = gym.make('MountainCar-v0').env
    runs = 30
    n_eps = 50
    alphas = np.arange(1, 8) / 4.0
    lambdas = [0.99, 0.95, 0.5, 0]
    epsilon = 0
    gamma = 1
    n_tilings = 8
    max_steps = 5000
    is_accumulated = False

    steps = np.zeros((len(lambdas), len(alphas), runs, n_eps))
    for lambda_idx in range(len(lambdas)):
        for alpha_idx in range(len(alphas)):
            for run in trange(runs):
                sarsa_lambda = SarsaLambda(env, lambdas[lambda_idx], alphas[alpha_idx],
                    gamma, epsilon, n_tilings, is_accumulated, max_steps)
                for ep in range(n_eps):
                    step = sarsa_lambda.run()
                    steps[lambda_idx, alpha_idx, run, ep] = step

    steps = np.mean(steps, axis=3)
    steps = np.mean(steps, axis=2)

    for lambda_idx in range(len(lambdas)):
        plt.plot(alphas, steps[lambda_idx, :], label='lambda = %s' \
            % (str(lambdas[lambda_idx])))
    plt.xlabel(r'$\alpha$ * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.ylim([180, 300])
    plt.legend(loc='upper right')

    plt.savefig('./mountain-car-sarsa-lambda-replacing-trace.png')
    plt.close()


def true_online_sarsa_lambda_plot():
    env = gym.make('MountainCar-v0').env
    n_eps = 1500
    alpha = 0.2
    lambda_ = 0.4
    epsilon = 0
    gamma = 0.9
    n_tilings = 8
    max_steps = 5000

    reward_list = []
    ave_reward_list = []

    true_online_sarsa_lambda = TrueOnlineSarsaLambda(env, lambda_, alpha, 
        gamma, epsilon, n_tilings, n_eps, max_steps)

    for ep in trange(n_eps):
        total_reward = true_online_sarsa_lambda.run(ep)

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
    plt.savefig('./mountain-car-true-online-sarsa-lambda.png')
    plt.close()


if __name__ == '__main__':
    # sarsa_lambda_plot()
    true_online_sarsa_lambda_plot()