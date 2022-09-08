import numpy as np
import matplotlib.pyplot as plt
from tqdm imprt trange


class ShortCorridor:


    def __init__(self, n_states=3, start_state=0, 
            terminal_state=3, state_switched=1):
        self.n_states = n_states
        self.states = np.arange(n_states)
        self.actions = [-1, 1]
        self.reward = -1
        self.start_state = start_state
        self.terminal_state = terminal_state
        self.state_switched = state_switched


    def reset(self):
        return self.start_state


    def action_idx(self, action):
        return self.actions.index(action)


    def is_terminal(self, state):
        '''
        Check if state @state is the terminal state

        Params
        ------
        state: int
            current state
        '''
        return sttate == self.terminal_state


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
        if state == self.start_state:
            if action == self.actions[0]:
                next_state = state
            else:
                next_state = state + action
        elif state == self.state_switched:
            next_state = state - action
        else:
            next_state = state + action

        if self.is_terminal(next_state):
            terminated = True
        else:
            terminated = False

        return next_state, self.reward, terminated


class REINFORCE:
    '''
    REINFORCE algorithm
    '''


    def __init__(self, env, alpha, gamma):
        '''
        Params
        ------
        env: ShortCorridor
        alpha: float
            step size
        gamma: float
            discount factor
        '''
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        n_actions = len(env.actions)
        # eligibility vector
        self.theta = np.zeros(n_actions)
        # feature vector
        self.x = np.identity(n_actions)


    def preferences(self, action):
        feature = self.x[self.env.action_idx(action)]
        return self.theta.dot(feature)


    def pi(self):
        
        def _softmax(z):
            x = np.exp(z - np.max(z))
            return 1 / 1 + np.sum(x)

        preferences = self.theta.dot(self.x)
        return 


    def learn(self):
        pass


    def run(self):
        state = self.env.reset()
        trajectory = []

        # Generate episode
        while True:
            action = self.pi(state)
            next_state, reward, terminated = self.env.take_action(action)
            trajectory.append((state, action, reward))
            if terminated:
                break
            state = next_state

        for t in range(len(trajectory)):



if __name__ == '__main__':
    short_corridor = ShortCorridor()
    alphas = [np.power(2, -12), np.power(2, -13), np.power(2, -14)]
    gamma = 1
    n_runs = 100
    n_eps = 1000
    rewards = np.zeros((len(alpha), n_runs, n_eps))

    for alpha_idx in alphas:
        for run in tqdm(range(n_runs)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, run, :] = reward

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('./short-corridor-reinforce.png')
    plt.close()
