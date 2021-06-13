import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

np.random.seed(1)

big_course = ['WWWWWWWWWWWWWWWWWW',
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

# Tiny course for debug

tiny_course = ['WWWWWW',
               'Woooo+',
               'Woooo+',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'W--WWW',]

class RaceTrack:

	def __init__(self, grid):
		self.NOISE = 0
		self.MAX_VELOCITY = 4
		self.MIN_VELOCITY = 0
		self.starting_line = []
		self.track = None
		self.car_position = None
		self._load_track(grid)
		self._generate_start_state()
		self.velocity = np.array([0, 0], dtype=np.int16)


	def reset(self):
		self._generate_start_state()
		self.velocity = np.array([0, 0], dtype=np.int16)


	def get_state(self):
		return self.car_position.copy(), self.velocity.copy()


	def _generate_start_state(self):
		index = np.random.choice(len(self.starting_line))
		self.car_position = np.array(self.starting_line[index])


	def take_action(self, action):
		if self.is_terminal():
			return 0
		self._update_state(action)
		return -1


	def _update_state(self, action):
		# update velocity
		# with probability of 0.1, keep the velocity unchanged
		if not np.random.binomial(1, 0.1):
			self.velocity += np.array(action, dtype=np.int16)
			self.velocity = np.minimum(self.velocity, self.MAX_VELOCITY)
			self.velocity = np.maximum(self.velocity, self.MIN_VELOCITY)

		# update car position
		for tstep in range(0, self.MAX_VELOCITY + 1):
			t = tstep / self.MAX_VELOCITY
			position = self.car_position + np.round(self.velocity * t).astype(np.int16)

			if self.track[position[0], position[1]] == -1:
				self.reset()
				return
			if self.track[position[0], position[1]] == 2:
				self.car_position = position
				self.velocity = np.array([0, 0], dtype=np.int16)
				return
		self.car_position = position


	def _load_track(self, grid):
		y_len, x_len = len(grid), len(grid[0])
		self.track = np.zeros((x_len, y_len), dtype=np.int16)
		for y in range(y_len):
			for x in range(x_len):
				pt = grid[y][x]
				if pt == 'W':
					self.track[x, y] = -1
				elif pt == 'o':
					self.track[x, y] = 1
				elif pt == '-':
					self.track[x, y] = 0
				else:
					self.track[x, y] = 2
		self.track = np.fliplr(self.track)
		for y in range(y_len):
			for x in range(x_len):
				if self.track[x, y] == 0:
					self.starting_line.append((x, y))

	def is_terminal(self):
		return self.track[self.car_position[0], self.car_position[1]] == 2


def behavior_policy(state):
	return [np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])]


def target_policy(state):
	pass


actions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]


def off_policy_MC_control(episodes, gamma, grid):
	x_len, y_len = len(grid[0]), len(grid)
	Q = np.zeros((x_len, y_len, 5, 5, 3, 3)) - 40
	C = np.zeros((x_len, y_len, 5, 5, 3, 3))
	pi = np.zeros((x_len, y_len, 5, 5, 1, 2), dtype=np.int16)
	track = RaceTrack(grid)
	epsilon = 0.1

	for ep in range(episodes):
		print('episode ', ep + 1)
		track.reset()
		trajectory = []
		while not track.is_terminal():
			state = track.get_state()
			s_x, s_y = state[0][0], state[0][1]
			s_vx, s_vy = state[1][0], state[1][1]
			if not np.random.binomial(1, epsilon):
				action = pi[s_x, s_y, s_vx, s_vy]
			else:
				action = behavior_policy(state)
			action = np.array(action).reshape((2,))
			reward = track.take_action(action)
			trajectory.append([state, action, reward])
		G = 0
		W = 1
		print(len(trajectory))
		while len(trajectory) > 0:
			state, action, reward = trajectory.pop()
			G = gamma * G + reward
			sp_x, sp_y, sv_x, sv_y = state[0][0], state[0][1], state[1][0], state[1][1]
			a_x, a_y = action
			s = (sp_x, sp_y, sv_x, sv_y)
			s_a = (sp_x, sp_y, sv_x, sv_y, a_x, a_y)
			C[s_a] += W
			Q[s_a] += W/C[s_a]*(G-Q[s_a])
			q_max = -1e5
			a_max = None
			for act in actions:
				sa_max = sp_x, sp_y, sv_x, sv_y, act[0], act[1]
				if Q[sa_max] > q_max:
					q_max = Q[sa_max]
					a_max = act
			pi[s] = a_max
			if np.array_equal(pi[s].reshape((2,)), action):
				break
			W *= 1/(1-8/9*epsilon)
	return pi
			

if __name__ == '__main__':
	gamma = 0.9
	episodes = 100000
	grid = big_course
	policy = off_policy_MC_control(episodes, gamma, grid)
	track_ = RaceTrack(grid)
	x_len, y_len = len(grid[0]), len(grid)
	pos_map = np.zeros((x_len, y_len))
	G = 0
	for e in range(1000):
		state = track_.get_state()
		s_x, s_y = state[0][0], state[0][1]
		s_vx, s_vy = state[1][0], state[1][1]
		pos_map[s_x, s_y] += 1
		action = policy[s_x, s_y, s_vx, s_vy]
		action = action.reshape((2, ))
		reward = track_.take_action(action)
		G += reward
		if track_.is_terminal():
			break  
	print('Sample trajectory on learned policy:')
	pos_map = (pos_map > 0).astype(np.float32)
	pos_map +=  track_.track  # overlay track course
	plt.imshow(np.flipud(pos_map.T), cmap='hot', interpolation='nearest')
	plt.show()

