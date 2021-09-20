import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(1)

big_track = ['WWWWWWWWWWWWWWWWWW',
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

class RaceTrack:

	def __init__(self, grid):
		self.NOISE = 0
		self.MAX_VELOCITY = 4
		self.MIN_VELOCITY = 0
		self.starting_line = []
		self.track = None
		self.car_position = None
		self.actions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
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
		# rotate the track in order to sync the track with actions
		self.track = np.fliplr(self.track)
		for y in range(y_len):
			for x in range(x_len):
				if self.track[x, y] == 0:
					self.starting_line.append((x, y))

	def is_terminal(self):
		return self.track[self.car_position[0], self.car_position[1]] == 2


def behavior_policy(track, state):
	index = np.random.choice(len(track.actions))
	return np.array(track.actions[index])


def off_policy_MC_control(episodes, gamma, grid):
	x_len, y_len = len(grid[0]), len(grid)
	Q = np.zeros((x_len, y_len, 5, 5, 3, 3)) - 40
	C = np.zeros((x_len, y_len, 5, 5, 3, 3))
	pi = np.zeros((x_len, y_len, 5, 5, 1, 2), dtype=np.int16)
	track = RaceTrack(grid)
	# for epsilon-soft greedy policy
	epsilon = 0.1

	for ep in tqdm(range(episodes)):
		# print('episode ', ep+1)
		track.reset()
		trajectory = []
		while not track.is_terminal():
			state = track.get_state()
			s_x, s_y = state[0][0], state[0][1]
			s_vx, s_vy = state[1][0], state[1][1]
			if not np.random.binomial(1, epsilon):
				action = pi[s_x, s_y, s_vx, s_vy, 0]
			else:
				action = behavior_policy(track, state)
			reward = track.take_action(action)
			trajectory.append([state, action, reward])
		G = 0
		W = 1
		while len(trajectory) > 0:
			state, action, reward = trajectory.pop()
			G = gamma * G + reward
			sp_x, sp_y, sv_x, sv_y = state[0][0], state[0][1], state[1][0], state[1][1]
			a_x, a_y = action
			s_a = (sp_x, sp_y, sv_x, sv_y, a_x, a_y)
			C[s_a] += W
			Q[s_a] += W/C[s_a]*(G-Q[s_a])
			q_max = -1e5
			a_max = None
			for act in track.actions:
				sa_max = sp_x, sp_y, sv_x, sv_y, act[0], act[1]
				if Q[sa_max] > q_max:
					q_max = Q[sa_max]
					a_max = act
			pi[sp_x, sp_y, sv_x, sv_y, 0] = a_max
			if not np.array_equal(pi[sp_x, sp_y, sv_x, sv_y, 0], action):
				break
			W *= 1 / (1 - epsilon + epsilon / 9)
	return pi
			

if __name__ == '__main__':
	gamma = 0.9
	episodes = 10000
	grid = big_track
	policy = off_policy_MC_control(episodes, gamma, grid)
	track_ = RaceTrack(grid)
	x_len, y_len = len(grid[0]), len(grid)
	trace = np.zeros((x_len, y_len))
	for _ in range(1000):
		state = track_.get_state()
		sp_x, sp_y, sv_x, sv_y = state[0][0], state[0][1], state[1][0], state[1][1]
		trace[sp_x, sp_y] += 1
		action = policy[sp_x, sp_y, sv_x, sv_y, 0]
		reward = track_.take_action(action)
		if track_.is_terminal():
			break
	trace = (trace > 0).astype(np.float32)
	trace += track_.track
	plt.imshow(np.flipud(trace.T))
	plt.savefig('./racetrack_off_policy_control.png')
	plt.close()


