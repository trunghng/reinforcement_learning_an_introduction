## Reinforcement Learning: An Introduction
Python Implementation for problems in the [**Reinforcement Learning: An Introduction**](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262193981) written by Richard S. Sutton & Andrew G. Barto.

Many ideas in the solutions are inspired by or taken from this well-known [repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) of Shangtong Zhang

### DETAILS
#### Chapter 3: Finite MDPs
- [x] Gridworld - **Bellman equation**: [source code](./chapter-03/gridworld.py), result (shown on the command prompt after running the code).
#### Chapter 4: Dynamic Programming
- [x] Gridworld - **Iterative Policy Evaluation**: [source code](./chapter-04/gridworld.py), result (shown on the command prompt after running the code).
- [x] Jack's Car Rental - **Policy Iteration**: [source code](./chapter-04/jackscar.py), [result](./chapter-04/jackscar.png).
- [x] Gambler's Problem - **Value Iteration**: [source code](./chapter-04/gambler.py), [result](./chapter-04/gambler.png).
#### Chapter 5: Monte Carlo Methods
- [x] Blackjack - **First-visit Monte Carlo**: [source code](./chapter-05/blackjack.py), [result](./chapter-05/blackjack_first_visit_MC.png).
- [x] Blackjack - **Monte Carlo Exploring Starts**: [source code](./chapter-05/blackjack.py), [result](./chapter-05/blackjack_monte_carlo_es.png).
- [x] Blackjack - **Off-policy Monte Carlo Prediction**: [source code](./chapter-05/blackjack.py), [result](./chapter-05/blackjack_monte_carlo_off_policy.png).
- [x] Infinite Variance - **OIS**: [source code](./chapter-05/infinite-variance.py), [result](./chapter-05/infinite_variance.png).
- [x] Racetrack - **Off-policy Monte Carlo Control**: [source code](./chapter-05/racetrack.py), [result](./chapter-05/racetrack_off_policy_control.png).
#### Chapter 6: TD Learning
- [x] Random Walk - **Constant-alpha Monte Carlo**: [source code](./chapter-06/random_walk.py), [result](./chapter-06/random_walk.png).
- [x] Random Walk - **TD(0) w/ & w/o Batch Updating**: [source code](./chapter-06/random_walk.py), [result](./chapter-06/random_walk_batch_updating.png).
- [x] Windy Gridworld - **Sarsa**: [source code](./chapter-06/windy_gridworld.py), [result](./chapter-06/windy_gridworld.png).
- [x] Cliff Walking - **Q-Learning**: [source code](./chapter-06/cliff_walking.py), [result 1](./chapter-06/cliff-walking-q-learning-sarsa.png), [result 2](./chapter-06/cliff-walking-q-learning-sarsa-expected-sarsa.png).
- [x] Cliff Walking - **Sarsa**: [source code](./chapter-06/cliff_walking.py), [result 1](./chapter-06/cliff-walking-q-learning-sarsa.png), [result 2](./chapter-06/cliff-walking-q-learning-sarsa-expected-sarsa.png).
- [x] Cliff Walking - **Expected Sarsa**: [source code](./chapter-06/cliff_walking.py), [result](./chapter-06/cliff-walking-q-learning-sarsa-expected-sarsa.png).
- [x] Moutain Car - **Q-Learning**: [source code](./chapter-06/mountain_car.py), [result](./chapter-06/mountain_car.png).
#### Chapter 7: n-step Bootstrapping
- [x] Random Walk - **n-step TD**: [source code](./chapter-07/random_walk.py), [result](./chapter-07/random_walk.png).
#### Chapter 8: Planning and Learning with Tabular Methods
- [x] Dyna Maze - **Dyna-Q**: [source code](./chapter-08/maze.py), [result](./chapter-08/dyna_maze.png).
- [x] Blocking Maze - **Dyna-Q**: [source code](./chapter-08/maze.py), [result](./chapter-08/blocking_maze.png).
- [x] Blocking Maze - **Dyna-Q+**: [source code](./chapter-08/maze.py), [result](./chapter-08/blocking_maze.png).
- [x] Shortcut Maze - **Dyna-Q**: [source code](./chapter-08/maze.py), [result](./chapter-08/shortcut_maze.png).
- [x] Shortcut Maze - **Dyna-Q+**: [source code](./chapter-08/maze.py), [result](./chapter-08/shortcut_maze.png).
- [x] Mazes - **Dyna-Q**: [source code](./chapter-08/maze.py), [result](./chapter-08/prioritized_sweeping.png).
- [x] Mazes - **Prioritized Sweeping**: [source code](./chapter-08/maze.py), [result](./chapter-08/prioritized_sweeping.png).
- [x] Task - **Trajectory Sampling**: [source code](./chapter-08/trajectory_sampling.py), [result](./chapter-08/trajectory_sampling.png).
#### Chapter 9: On-policy Prediction with Approximation
- [x] Random Walk - **Gradient Monte Carlo w/ State Aggregation**: [source code](./chapter-09/random_walk.py), [result](./chapter-09/gradient_mc_state_agg.png).
- [x] Random Walk - **Semi-gradient TD**: [source code](./chapter-09/random_walk.py), [result](./chapter-09/semi_gradient_td.png).
- [x] Random Walk - **Gradient Monte Carlo w/ Fourier basis**: [source code](./chapter-09/random_walk.py), [result](./chapter-09/gradient_mc_bases.png).
- [x] Random Walk - **Gradient Monte Carlo w/ Polynomial basis**: [source code](./chapter-09/random_walk.py), [result](./chapter-09/gradient_mc_bases.png).
- [x] Random Walk - **Gradient Monte Carlo w/ Tile Coding**: [source code](./chapter-09/random_walk.py), [result](./chapter-09/gradient_mc_tile_coding.png).
- [x] Square Wave - **Linear function approximation w/ Coarse Coding**: [source code](./chapter-09/square_wave.py), [result](./chapter-09/squave_wave_function.png).
#### Chapter 10: On-policy Control with Approximation
- [x] Mountain Car - **Episodic Semi-gradient Sarsa**: [source code](./chapter-10/mountain_car.py), [result](./chapter-10/mountain-car-ep-semi-grad-sarsa.png).
- [x] Mountain Car - **Episodic Semi-gradient n-step Sarsa**: [source code](./chapter-10/mountain_car.py), [result](./chapter-10/mountain-car-ep-semi-grad-n-step-sarsa.png).
- [ ] Access Control - **Differential Semi-gradient Sarsa**: [source code](./chapter-10/access_control.py), [result](#).
#### Chapter 11: Off-policy Methods with Approximation
#### Chapter 12: Eligible Traces
- [x] Random Walk - **Offline lambda-return**: [source code](./chapter-12/random_walk.py), [result](./chapter-12/random-walk-offline-lambda-return.png).
- [x] Random Walk - **TD(lambda)**: [source code](./chapter-12/random_walk.py), [result](./chapter-12/random-walk-td-lambda.png).
- [x] Random Walk - **True online TD(lambda)**: [source code](./chapter-12/random_walk.py), [result](./chapter-12/random-walk-true-online-td-lambda.png).
- [x] Mountain Car - **Sarsa(lambda) w/ replacing traces**: [source code](./chapter-12/mountain_car.py), [result](./chapter-12/mountain-car-sarsa-lambda-replacing-trace.png).
- [x] Mountain Car - **Sarsa(lambda) w/ accumulating traces**: [source code](./chapter-12/mountain_car.py), [result](#).
- [x] Mountain Car - **True online Sarsa(lambda)**: [source code](./chapter-12/mountain_car.py), [result](./chapter-12/mountain-car-true-online-sarsa-lambda.png).
#### Chapter 13: Policy Gradient
- [x] Short Corridor - **REINFORCE**: [source code](./chapter-13/short_corridor.py), [result](./chapter-13/short-corridor-reinforce.png).
- [x] Short Corridor - **REINFORCE w/ baseline**: [source code](./chapter-13/short_corridor.py), [result](./chapter-13/short-corridor-reinforce-baseline.png).
