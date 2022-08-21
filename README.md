# Reinforcement Learning: An Introduction
Python Implementation for problems in the *Reinforcement Learning: An Introduction - Richard S. Sutton and Andrew G. Barto*  

Many ideas in the solutions are inspired by or taken from this well-known [repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) of Shangtong Zhang

### DETAILS
#### Chapter 3: Finite MDPs
- [x] Gridworld - **Bellman equation**
	- Code: [link](./chapter-3/gridworld.py)
	- Result: shown on the command prompt after running the code
#### Chapter 4: Dynamic Programming
- [x] Gridworld - **Iterative Policy Evaluation**
	- Code: [link](./chapter-4/gridworld.py)
	- Result: shown on the command prompt after running the code
- [x] Jack's Car Rental - **Policy Iteration**
	- Code: [link](./chapter-4/jackscar.py)
	- Result: 
- [x] Gambler's Problem - **Value Iteration**
	- Code: [link](./chapter-4/gambler.py)
	- Result: [link](./chapter-4/gambler.png)
#### Chapter 5: Monte Carlo Methods
- [x] Blackjack - **First-visit Monte Carlo**
	- Code: [link](./chapter-5/blackjack.py)
	- Result: [link](./chapter-5/blackjack_first_visit_MC.png)
- [x] Blackjack - **Monte Carlo Exploring Starts**
	- Code: [link](./chapter-5/blackjack.py)
	- Result: [link](./chapter-5/blackjack_monte_carlo_es.png)
- [x] Blackjack - **Off-policy Monte Carlo Prediction**
	- Code: [link](./chapter-5/blackjack.py)
	- Result: [link](./chapter-5/blackjack_monte_carlo_off_policy.png)
- [x] Infinite Variance - **OIS**
	- Code: [link](./chapter-5/infinite-variance.py)
	- Result: [link](./chapter-5/infinite_variance.png)
- [x] Racetrack - **Off-policy Monte Carlo Control**
	- Code: [link](./chapter-5/racetrack.py)
	- Result: [link](./chapter-5/racetrack_off_policy_control.png)
#### Chapter 6: TD Learning
- [x] Random Walk - **Constant-alpha Monte Carlo**
	- Code: [link](./chapter-6/random_walk.py)
	- Result: [link](./chapter-6/random_walk.png)
- [x] Random Walk - **TD(0) w/ & w/o Batch Updating**
	- Code: [link](./chapter-6/random_walk.py)
	- Result: [link](./chapter-6/random_walk_batch_updating.png)
- [x] Windy Gridworld - **Sarsa**
	- Code: [link](./chapter-6/windy_gridworld.py)
	- Result: [link](./chapter-6/windy_gridworld.png)
- [x] Cliff Walking - **Q-Learning**
	- Code: [link](./chapter-6/cliff_walking.py)
	- Result: [link](./chapter-6/cliff_walking.png)
- [x] Cliff Walking - **Sarsa**
	- Code: [link](./chapter-6/cliff_walking.py)
	- Result: [link](./chapter-6/cliff_walking.png)
- [ ] Cliff Walking - **Expected Sarsa**
#### Chapter 7: n-step Bootstrapping
- [x] Random Walk - **n-step TD**
	- Code: [link](./chapter-7/random_walk.py)
	- Result: [link](./chapter-7/random_walk.png)
#### Chapter 8: Planning and Learning with Tabular Methods
- [x] Dyna Maze - **Dyna-Q**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/dyna_maze.png)
- [x] Blocking Maze - **Dyna-Q**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/blocking_maze.png)
- [x] Blocking Maze - **Dyna-Q+**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/blocking_maze.png)
- [x] Shortcut Maze - **Dyna-Q**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/shortcut_maze.png)
- [x] Shortcut Maze - **Dyna-Q+**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/shortcut_maze.png)
- [x] Mazes - **Dyna-Q**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/prioritized_sweeping.png)
- [x] Mazes - **Prioritized Sweeping**
	- Code: [link](./chapter-8/maze.py)
	- Result: [link](./chapter-8/prioritized_sweeping.png)
- [x] Trajectory Sampling
	- Code: [link](./chapter-8/trajectory_sampling.py)
	- Result: [link](./chapter-8/trajectory_sampling.png)
#### Chapter 9: On-policy Prediction with Approximation
- [x] Random Walk - **Gradient Monte Carlo w/ State Aggregation**
	- Code: [link](./chapter-9/random_walk.py)
	- Result: [link](./chapter-9/gradient_mc_state_agg.png)
- [x] Random Walk - **Semi-gradient TD**
	- Code: [link](./chapter-9/random_walk.py)
	- Result: [link](./chapter-9/semi_gradient_td.png)
- [x] Random Walk - **Gradient Monte Carlo w/ Fourier basis**
	- Code: [link](./chapter-9/random_walk.py)
	- Result: [link](./chapter-9/gradient_mc_bases.png)
- [x] Random Walk - **Gradient Monte Carlo w/ Polynomial basis**
	- Code: [link](./chapter-9/random_walk.py)
	- Result: [link](./chapter-9/gradient_mc_bases.png)
- [x] Random Walk - **Gradient Monte Carlo w/ Tile Coding**
	- Code: [link](./chapter-9/random_walk.py)
	- Result: [link](./chapter-9/gradient_mc_tile_coding.png)
- [x] Square Wave - **Linear function approximation w/ Coarse Coding**
	- Code: [link](./chapter-9/square_wave.py)
	- Result: [link](./chapter-9/squave_wave_function.png)
#### Chapter 10: On-policy Control with Approximation
- [ ] Mountain Car - **Episodic Semi-gradient Sarsa**
- [ ] Access Control - 
#### Chapter 11: Off-policy Methods with Approximation
#### Chapter 12: Eligible Traces
- [x] Random Walk - **Offline lambda-return**
	- Code: [link](./chapter-12/random_walk.py)
	- Result: [link](./chapter-12/offline-lambda-return.png)
- [x] Random Walk - **TD(lambda)**
	- Code: [link](./chapter-12/random_walk.py)
	- Result: [link](./chapter-12/td-lambda.png)
- [x] Random Walk - **True online TD(lambda)**
	- Code: [link](./chapter-12/random_walk.py)
	- Result: 
- [ ] Mountain Car - **Sarsa(lambda) w/ replacing traces**
- [ ] Mountain Car - **Sarsa(lambda) w/ accumulating traces**
- [ ] Mountain Car - **True online Sarsa(lambda)**
(Updating)