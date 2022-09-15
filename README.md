## Reinforcement Learning: An Introduction
Python Implementation for problems in the *Reinforcement Learning: An Introduction - Richard S. Sutton and Andrew G. Barto*  

Many ideas in the solutions are inspired by or taken from this well-known [repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) of Shangtong Zhang

### DETAILS
#### Chapter 3: Finite MDPs
- [x] Gridworld - **Bellman equation**
	- Code: [view](./chapter-03/gridworld.py)
	- Result: shown on the command prompt after running the code
#### Chapter 4: Dynamic Programming
- [x] Gridworld - **Iterative Policy Evaluation**
	- Code: [view](./chapter-04/gridworld.py)
	- Result: shown on the command prompt after running the code
- [x] Jack's Car Rental - **Policy Iteration**
	- Code: [view](./chapter-04/jackscar.py)
	- Result: [view](./chapter-04/jackscar.png)
- [x] Gambler's Problem - **Value Iteration**
	- Code: [view](./chapter-04/gambler.py)
	- Result: [view](./chapter-04/gambler.png)
#### Chapter 5: Monte Carlo Methods
- [x] Blackjack - **First-visit Monte Carlo**
	- Code: [view](./chapter-05/blackjack.py)
	- Result: [view](./chapter-05/blackjack_first_visit_MC.png)
- [x] Blackjack - **Monte Carlo Exploring Starts**
	- Code: [view](./chapter-05/blackjack.py)
	- Result: [view](./chapter-05/blackjack_monte_carlo_es.png)
- [x] Blackjack - **Off-policy Monte Carlo Prediction**
	- Code: [view](./chapter-05/blackjack.py)
	- Result: [view](./chapter-05/blackjack_monte_carlo_off_policy.png)
- [x] Infinite Variance - **OIS**
	- Code: [view](./chapter-05/infinite-variance.py)
	- Result: [view](./chapter-05/infinite_variance.png)
- [x] Racetrack - **Off-policy Monte Carlo Control**
	- Code: [view](./chapter-05/racetrack.py)
	- Result: [view](./chapter-05/racetrack_off_policy_control.png)
#### Chapter 6: TD Learning
- [x] Random Walk - **Constant-alpha Monte Carlo**
	- Code: [view](./chapter-06/random_walk.py)
	- Result: [view](./chapter-06/random_walk.png)
- [x] Random Walk - **TD(0) w/ & w/o Batch Updating**
	- Code: [view](./chapter-06/random_walk.py)
	- Result: [view](./chapter-06/random_walk_batch_updating.png)
- [x] Windy Gridworld - **Sarsa**
	- Code: [view](./chapter-06/windy_gridworld.py)
	- Result: [view](./chapter-06/windy_gridworld.png)
- [x] Cliff Walking - **Q-Learning**
	- Code: [view](./chapter-06/cliff_walking.py)
	- Result 1: [view](./chapter-06/cliff-walking-q-learning-sarsa.png)
	- Result 2: [view](./chapter-06/cliff-walking-q-learning-sarsa-expected-sarsa.png)
- [x] Cliff Walking - **Sarsa**
	- Code: [view](./chapter-06/cliff_walking.py)
	- Result 1: [view](./chapter-06/cliff-walking-q-learning-sarsa.png)
	- Result 2: [view](./chapter-06/cliff-walking-q-learning-sarsa-expected-sarsa.png)
- [x] Cliff Walking - **Expected Sarsa**
	- Code: [view](./chapter-06/cliff_walking.py)
	- Result: [view](./chapter-06/cliff-walking-q-learning-sarsa-expected-sarsa.png)
- [x] Moutain Car - **Q-Learning**
	- Code: [view](./chapter-06/mountain_car.py)
	- Result: [view](./chapter-06/mountain_car.png)
#### Chapter 7: n-step Bootstrapping
- [x] Random Walk - **n-step TD**
	- Code: [view](./chapter-07/random_walk.py)
	- Result: [view](./chapter-07/random_walk.png)
#### Chapter 8: Planning and Learning with Tabular Methods
- [x] Dyna Maze - **Dyna-Q**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/dyna_maze.png)
- [x] Blocking Maze - **Dyna-Q**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/blocking_maze.png)
- [x] Blocking Maze - **Dyna-Q+**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/blocking_maze.png)
- [x] Shortcut Maze - **Dyna-Q**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/shortcut_maze.png)
- [x] Shortcut Maze - **Dyna-Q+**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/shortcut_maze.png)
- [x] Mazes - **Dyna-Q**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/prioritized_sweeping.png)
- [x] Mazes - **Prioritized Sweeping**
	- Code: [view](./chapter-08/maze.py)
	- Result: [view](./chapter-08/prioritized_sweeping.png)
- [x] Trajectory Sampling
	- Code: [view](./chapter-08/trajectory_sampling.py)
	- Result: [view](./chapter-08/trajectory_sampling.png)
#### Chapter 9: On-policy Prediction with Approximation
- [x] Random Walk - **Gradient Monte Carlo w/ State Aggregation**
	- Code: [view](./chapter-09/random_walk.py)
	- Result: [view](./chapter-09/gradient_mc_state_agg.png)
- [x] Random Walk - **Semi-gradient TD**
	- Code: [view](./chapter-09/random_walk.py)
	- Result: [view](./chapter-09/semi_gradient_td.png)
- [x] Random Walk - **Gradient Monte Carlo w/ Fourier basis**
	- Code: [view](./chapter-09/random_walk.py)
	- Result: [view](./chapter-09/gradient_mc_bases.png)
- [x] Random Walk - **Gradient Monte Carlo w/ Polynomial basis**
	- Code: [view](./chapter-09/random_walk.py)
	- Result: [view](./chapter-09/gradient_mc_bases.png)
- [x] Random Walk - **Gradient Monte Carlo w/ Tile Coding**
	- Code: [view](./chapter-09/random_walk.py)
	- Result: [view](./chapter-09/gradient_mc_tile_coding.png)
- [x] Square Wave - **Linear function approximation w/ Coarse Coding**
	- Code: [view](./chapter-09/square_wave.py)
	- Result: [view](./chapter-09/squave_wave_function.png)
#### Chapter 10: On-policy Control with Approximation
- [x] Mountain Car - **Episodic Semi-gradient Sarsa**
	- Code: [view](./chapter-10/mountain_car.py)
	- Result: [view](./chapter-10/mountain-car-ep-semi-grad-sarsa.png)
- [x] Mountain Car - **Episodic Semi-gradient n-step Sarsa**
	- Code: [view](./chapter-10/mountain_car.py)
	- Result: [view](./chapter-10/mountain-car-ep-semi-grad-n-step-sarsa.png)
- [ ] Access Control - **Differential Semi-gradient Sarsa**
	- Code: [view](./chapter-10/access_control.py)
	- Result: 
#### Chapter 11: Off-policy Methods with Approximation
#### Chapter 12: Eligible Traces
- [x] Random Walk - **Offline lambda-return**
	- Code: [view](./chapter-12/random_walk.py)
	- Result: [view](./chapter-12/random-walk-offline-lambda-return.png)
- [x] Random Walk - **TD(lambda)**
	- Code: [view](./chapter-12/random_walk.py)
	- Result: [view](./chapter-12/random-walk-td-lambda.png)
- [x] Random Walk - **True online TD(lambda)**
	- Code: [view](./chapter-12/random_walk.py)
	- Result: [view](./chapter-12/random-walk-true-online-td-lambda.png)
- [x] Mountain Car - **Sarsa(lambda) w/ replacing traces**
	- Code: [view](./chapter-12/mountain_car.py)
	- Result: [view](./chapter-12/mountain-car-sarsa-lambda-replacing-trace.png)
- [x] Mountain Car - **Sarsa(lambda) w/ accumulating traces**
	- Code: [view](./chapter-12/mountain_car.py)
	- Result:
- [x] Mountain Car - **True online Sarsa(lambda)**
	- Code: [view](./chapter-12/mountain_car.py)
	- Result: [view](./chapter-12/mountain-car-true-online-sarsa-lambda.png)
#### Chapter 13: Policy Gradient
- [ ] Short Corridor - **REINFORCE**
	- Code: [view](./chapter-13/short_corridor.py)
	- Result: [view](./chapter-13/short-corridor-reinforce.png)
- [ ] Short Corridor - **REINFORCE w/ baseline**
	- Code: [view](./chapter-13/short_corridor.py)
	- Result: [view](./chapter-13/short-corridor-reinforce-baseline.png)
