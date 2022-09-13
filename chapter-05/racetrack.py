import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env import RaceTrack

np.random.seed(1)

class OffPolicyMonteCarloControl:
	'''
	Off-policy Monte Carlo control using weighted IS agent

	Q: state-action value function
	C_n = C(S_n, A_n): cumulative sum of the weights given to the first n returns

	G += gamma * R_{t+1}
	C(S_t, A_t) += W
	Q(S_t, A_t) += W / C(S_t, A_t) * [G - Q(S_t, A_t)]
	W *= 1 / b(A_t|S_t)
	'''

	def __init__(self, env: RaceTrack, 
			gamma: float, epsilon: float,
			n_eps: int) -> None:
		'''
		Params
		------
		env: RaceTrack env
		gamma: discount factor
		epsilon: exloration param
		n_eps: number of episodes
		'''
		self.env = env
		self.gamma = gamma
		self.epsilon = epsilon
		self.n_eps = n_eps
		position_x_dim, position_y_dim = env.position_space.shape
		velocity_x_dim, velocity_y_dim, _ = env.velocity_space.shape
		action_size = len(env.action_space)
		self.Q = np.zeros((position_x_dim, position_y_dim, 
			velocity_x_dim, velocity_y_dim, action_size)) - 40
		self.C = np.zeros((position_x_dim, position_y_dim, 
			velocity_x_dim, velocity_y_dim, action_size))
		self.pi = np.random.randint(action_size, size=(position_x_dim, 
			position_y_dim, velocity_x_dim, velocity_y_dim), dtype=np.int16)


	def _reset(self) -> np.ndarray:
		'''
		Reset agent
		'''
		return self.env.reset()


	def _behavior_policy(self, state) -> Tuple[int, int]:
		'''
		Epsilon-greedy policy

		Return
		------
		action: chosen action
		'''
		if np.random.binomial(1, self.epsilon):
			action = np.random.choice(self.env.action_space)
		else:
			action = self._greedy_policy(state)
		
		return action


	def _greedy_policy(self, state) -> Tuple[int, int]:
		'''
		Greedy policy

		Return
		------
		action: chosen action
		'''
		p_x, p_y, v_x, v_y = state[0, 0], state[0, 1], \
							state[1, 0], state[1, 1]
		max_value = self.Q[p_x, p_y, v_x, v_y, :].max()
		action = np.random.choice(np.flatnonzero(
			self.Q[p_x, p_y, v_x, v_y, :] == max_value))

		return action


	def _run_episode(self) -> None:
		'''
		Perform an episode
		'''
		state = self._reset()
		trajectory = []

		while True:
			action = self._behavior_policy(state)
			next_state, reward, terminated = self.env.step(action)
			trajectory.append((state, action, reward))
			state = next_state

			if terminated:
				break

		G = 0
		W = 1

		while len(trajectory) > 0:
			state, action, reward = trajectory.pop()
			G = self.gamma * G + reward

			p_x, p_y, v_x, v_y = state[0, 0], state[0, 1], \
							state[1, 0], state[1, 1]

			self.C[p_x, p_y, v_x, v_y, action] += W
			self.Q[p_x, p_y, v_x, v_y, action] += W / self.C[p_x, p_y, v_x, \
				v_y, action] * (G - self.Q[p_x, p_y, v_x, v_y, action])

			self.pi[p_x, p_y, v_x, v_y] = self._greedy_policy(state)

			if action != self.pi[p_x, p_y, v_x, v_y]:
				break

			W += 1 / (1 - self.epsilon + self.epsilon / 9)


	def run(self) -> np.ndarray:
		for ep in trange(self.n_eps):
			self._run_episode()

		return self.pi


if __name__ == '__main__':
	track = ['WWWWWWWWWWWWWWWWWW',
			'WWWWooooooooooooo+',
			'WWWoooooooooooooo+',
			'WWWoooooooooooooo+',
			'WWooooooooooooooo+',
			'Woooooooooooooooo+',
			'Woooooooooooooooo+',
			'WooooooooooWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WoooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWooooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWoooooooWWWWWWWW',
			'WWWWooooooWWWWWWWW',
			'WWWWooooooWWWWWWWW',
			'WWWW------WWWWWWWW']

	track2 = ['WWW+++++++WWWWWWWW',
		'WWWoooooooWWWWWWWW',
		'WWWoooooooWWWWWWWW',
		'WWWoooooooWWWWWWWW',
		'WWWoooooooWWWWWWWW',
		'WWWoooooooWWWWWWWW',
		'WWWoooooooWWWWWWWW',
		'WWWWooooooWWWWWWWW',
		'WWWWooooooWWWWWWWW',
		'WWWW------WWWWWWWW']
	          
	velocity_unchanged_prob = 0.1
	env = RaceTrack(track)
	gamma = 0.9
	epsilon = 0.1
	n_eps = 10000
	off_policy_mc_control = OffPolicyMonteCarloControl(env, gamma, epsilon, n_eps)
	policy = off_policy_mc_control.run()

	trace = np.zeros((policy.shape[0], policy.shape[1]))

	state = env.reset()
	for _ in range(1000):
		p_x, p_y, v_x, v_y = state[0, 0], state[0, 1], state[1, 0], state[1, 1]
		trace[p_x, p_y] += 1
		action = policy[p_x, p_y, v_x, v_y]
		next_state, reward, terminated = env.step(action)
		state = next_state

		if terminated:
			break

	trace = (trace > 0).astype(np.float32)
	trace += env.position_space

	plt.imshow(np.flipud(trace.T))
	plt.savefig('./racetrack_off_policy_control.png')
	plt.close()
