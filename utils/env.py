from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np


class Env(ABC):
    '''
    Env abstract class
    '''

    def __init__(self):
        pass


    @abstractmethod
    def reset(self):
        pass


    @abstractmethod
    def _terminated(self):
        pass


    @abstractmethod
    def _get_reward(self):
        pass


    @abstractmethod
    def step(self):
        pass


class RandomWalk(Env):
    '''
    Random walk env
    '''

    def __init__(self, n_states: int, 
                start_state: int,
                terminal_states: List[int], 
                action_space: List[int]=[-1, 1],
                reward_space: List[float]=[1, 0, -1],
                transition_probs: Dict[int, float]={-1: 0.5, 1: 0.5},
                transition_radius: int=None) -> None:
        '''
        Params
        ------
        n_states: number of state
        start_state: start state
        terminal_states: list of terminal states
        action_space: action space
        reward_space: reward space
        transition_probs: transition probabilities
        '''
        self.n_states = n_states
        self.state_space = np.arange(1, n_states + 1)
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.action_space = action_space
        self.reward_space = reward_space
        self.transition_probs = transition_probs
        self.transition_radius = transition_radius


    def reset(self) -> int:
        '''
        Reset env
        '''
        self.state = self.start_state
        return np.copy(self.state)


    def _terminated(self) -> bool:
        '''
        Whether agent is in terminal state
        '''
        return self.state in self.terminal_states


    def _get_reward(self) -> float:
        '''
        Get reward

        Returns
        -------
        reward: reward corresponding
        '''
        if self.state == self.terminal_states[1]:
            reward = self.reward_space[0]
        elif self.state == self.terminal_states[0]:
            reward = self.reward_space[-1]
        else:
            reward = self.reward_space[1]
        return reward


    def step(self, action: int, 
            state: int=None) -> Tuple[int, float, bool]:

        '''
        Take action

        Params
        ------
        action: action taken
        state: state of the agent

        Returns
        -------
        next_state: next state
        reward: corresponding reward
        reward: whether next state is a terminal state
        '''
        assert action in self.action_space, 'Invalid action!'
        if state is not None:
            self.state = state

        if self.transition_radius is None:
            next_state = self.state + action
        else:
            step = np.random.randint(1, self.transition_radius + 1)
            next_state = min(self.terminal_states[1], 
                max(self.terminal_states[0], self.state + action * step))

        self.state = next_state
        reward = self._get_reward()
        terminated = self._terminated()
        return next_state, reward, terminated


    def get_state_transition(self, state: int, 
                            action: int) -> Dict[int, float]:
        '''
        Get state transition at state @state

        Params
        ------
        state: state of the agent
        action: action taken at state @state

        Returns
        -------
        state_transition: state transition probabilities
        '''
        assert self.transition_radius is not None, 'Transition radius required!'
        def __possible_next_states(state: int, action: int) -> np.ndarray:
            if action == self.action_space[0]:
                next_states = np.arange(max(self.terminal_states[0], state - 
                    self.transition_radius), state + action + 1)
            elif action == self.action_space[1]:
                next_states = np.arange(state + action, 
                    min(self.terminal_states[1], state + self.transition_radius) + 1)
            return next_states


        next_states = __possible_next_states(state, action)
        next_state_prob = 1.0 / self.transition_radius
        state_transition = {next_state: next_state_prob for \
            next_state in next_states}

        if self.terminal_states[0] == next_states[0]:
            state_transition[next_states[0]] += (self.transition_radius 
                - len(next_states)) * next_state_prob
        elif self.terminal_states[1] == next_states[-1]:
            state_transition[next_states[-1]] += (self.transition_radius 
                - len(next_states)) * next_state_prob

        return state_transition


class RaceTrack(Env):
    '''
    Race track env
    '''

    def __init__(self, track: List[str], 
                velocity_unchanged_prob: float=None) -> None:
        '''
        Params
        ------
        track: raw track
        velocity_unchanged_prob: velocity unchanged probability
        '''
        self.position_space, self.starting_line, \
            self.finish_line = self._load_track(track)
        self.max_position = np.array(self.position_space.shape) - 1
        self.min_position = np.array([0, 0])
        self.max_velocity = 4
        self.min_velocity = 0
        self.velocity_space = np.array([[(i, j) 
            for i in range(self.min_velocity, self.max_velocity + 1)] 
            for j in range(self.min_velocity, self.max_velocity + 1)])
        self.action_space_ = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                            (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.action_space = list(range(len(self.action_space_)))
        self.velocity_unchanged_prob = velocity_unchanged_prob


    def _load_track(self, track: List[str]) -> Tuple[np.ndarray, \
            List[Tuple[int, int]], List[Tuple[int, int]]]:
        '''
        Load raw track

        Returns
        -------
        track_: race track
        starting_line: starting line
        finish_line: finish line
        '''
        y_len, x_len = len(track), len(track[0])
        track_ = np.zeros((x_len, y_len))
        starting_line = []
        finish_line = []

        for y in range(y_len):
            for x in range(x_len):
                pt = track[y][x]
                if pt == 'W':
                    track_[x, y] = -1
                elif pt == 'o':
                    track_[x, y] = 1
                elif pt == '-':
                    track_[x, y] = 0
                else:
                    track_[x, y] = 2
        # rotate the track in order to sync the track with actions
        track_ = np.fliplr(track_)
        for y in range(y_len):
            for x in range(x_len):
                if track_[x, y] == 0:
                    starting_line.append((x, y))
                elif track_[x, y] == 2:
                    finish_line.append((x, y))

        return track_, starting_line, finish_line


    def reset(self) -> np.ndarray:
        '''
        Reset the car
        '''
        index = np.random.choice(len(self.starting_line))
        position = self.starting_line[index]
        velocity = [0, 0]
        self.state = np.array([position, velocity])
        return np.copy(self.state)


    def _terminated(self) -> bool:
        '''
        Whether the car has reached the finish line
        '''
        position = self.state[0]
        return tuple(position) in self.finish_line


    def _hit_wall(self) -> bool:
        '''
        Whether the car has hit the wall
        '''
        position = self.state[0]
        return self.position_space[position[0], position[1]] == -1


    def _get_reward(self) -> float:
        return -1.0


    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool]:
        '''
        Take action

        Returns
        -------
        next_state: next state
        reward: corresponding reward
        terminated: whether the car has finished the track
        '''
        assert action in self.action_space, 'Invalid action!'

        position, velocity = self.state[0], self.state[1]

        if (self.velocity_unchanged_prob is not None and not \
            np.random.binomial(1, self.velocity_unchanged_prob)) \
            or self.velocity_unchanged_prob is None:
            action_ = self.action_space_[action]
            velocity += np.array(action_)
            velocity = np.minimum(velocity, self.max_velocity)
            velocity = np.maximum(velocity, self.min_velocity)

        next_position = position + velocity
        next_position = np.minimum(next_position, self.max_position)
        next_position = np.maximum(next_position, self.min_position)
        self.state = np.array([next_position, velocity])

        if self._terminated():
            terminated = True
        else:
            terminated = False
            if self._hit_wall():
                self.reset()

        reward = self._get_reward()
        next_state = np.copy(self.state)

        return next_state, reward, terminated


class GridWorld(Env):
    '''
    Gridworld env
    '''

    def __init__(self, height: int, 
                width: int, 
                start_state: Tuple[int, int]=None, 
                terminal_states: List[Tuple[int, int]]=None,
                transition_probs: List[float]=[0.25, 0.25, 0.25, 0.25],
                wind_dist: List[int]=None,
                cliff: List[Tuple[int, int]]=None,
                obstacles: List[Tuple[int, int]]=None,
                special_states: Tuple[List[Tuple[int, int]], \
                    List[Tuple[int, int]], List[float]]=[None, None, None]) -> None:
        '''
        Params
        ------
        height: vertical length of the grid
        width: horizontal length of the grid
        start_state: start state
        terminal_states: list of terminal states
        transition_probs: transition probabilities
        wind_dist: wind distribution
        cliff: cliff region
        obstacles: obstacles
        special_states: special states, along with corresponding next states, rewards
        '''
        self.height = height
        self.width = width
        self.start_state = np.array(start_state)
        self.terminal_states = [np.array(state) for state in terminal_states]
        self.state_space = [np.array([i, j]) for i in range(height) 
                                for j in range(width)]
        self.action_space_ = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_space = list(range(len(self.action_space_)))
        self.high = np.array([height, width]) - 1
        self.low = np.array([0, 0])
        self.transition_probs = transition_probs
        self.wind_dist = wind_dist
        self.cliff = cliff
        self.obstacles = obstacles
        self.states_, self.next_states_, self.rewards_ = special_states


    def reset(self) -> None:
        self.state = self.start_state
        return np.copy(self.state)


    def _terminated(self) -> bool:
        for state in self.terminal_states:
            if (self.state == state).all():
                return True
        return False


    def terminated(self, state) -> bool:
        for state_ in self.terminal_states:
            if (state == state_).all():
                return True
        return False


    def _get_reward(self) -> float:
        return -1.0


    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        '''
        Take action

        Returns
        -------
        next_state: next state
        reward: corresponding reward
        terminated: whether the agent has reached the terminal state
        '''
        assert action in self.action_space, 'Invalid action!'

        state_ = (self.state[0], self.state[1])
        if self.states_ is not None and state_ in self.states_:
            index = self.states_.index(state_)
            next_state = np.array(self.next_states_[index])
            reward = self.rewards_[index]
        else:
            action_ = self.action_space_[action]
            next_state = self.state + np.array(action_)
            if self.wind_dist is not None:
                next_state += np.array([self.wind_dist[self.state[1]], 0])

            next_state = np.minimum(next_state, self.high)
            next_state = np.maximum(next_state, self.low)

            reward = self._get_reward()
            
            if self.cliff is not None and \
                (next_state[0], next_state[1]) in self.cliff:
                next_state = self.reset()
                reward = -100.0
            elif self.obstacles is not None:
                reward = 0

                if (next_state[0], next_state[1]) in self.obstacles:
                    next_state = self.state

        self.state = next_state
        terminated = self._terminated()

        if terminated and self.obstacles is not None:
            reward = 1.0

        return next_state, reward, terminated


    def set_obstacles(self, obstacles: List[Tuple[int, int]]) -> None:
        '''
        Set obstacles
        '''
        self.obstacles = obstacles


    def _extend(self, state: np.ndarray, resolution: int) -> List[Tuple[int, int]]:
        '''
        Extend state

        Params
        ------
        state: state of the agent
        resolution: extension param
        '''
        states = []
        x, y = state[0], state[1]

        for i in range(x * resolution, (x + 1) * resolution):
            for j in range(y * resolution, (y + 1) * resolution):
                states.append((i, j))
        return states


    def extend(self, resolution: int) -> object:
        '''
        Extend gridworld

        Params
        ------
        resolution: extension param

        Returns
        -------
        new gridworld
        '''
        height = self.height * resolution
        width = self.width * resolution
        start_state = self.start_state * resolution
        terminal_states = [np.array(state) for state_ in self.terminal_states \
            for state in self._extend(state_, resolution)]
        obstacles = [state for obstacle in self.obstacles \
            for state in self._extend(obstacle, resolution)]

        return GridWorld(height, width, start_state, 
            terminal_states, obstacles=obstacles)


class Gambler(Env):
    '''
    Gambler's problem env
    '''

    def __init__(self, goal: int) -> None:
        '''
        Params
        ------
        goal: goal of the gambler
        '''
        self.state_space = np.arange(goal + 1)
        self.terminal_states = [0, goal]
        self.high = goal
        self.low = 0


    def reset(self):
        pass


    def _terminated(self, state: int) -> bool:
        return state in self.terminal_states


    def _get_reward(self) -> float:
        return 0


    def action_space(self, state: int) -> np.ndarray:
        return np.arange(min(state, self.high - state) + 1)


    def step(self, state: int, action: int, head: bool) -> Tuple[int, float, bool]:
        assert action in self.action_space(state), 'Invalid action!'

        next_state = state + np.power(-1, head) * action
        terminated = self._terminated(state)

        if next_state == self.high:
            reward = 1
        else:
            reward = self._get_reward()

        return next_state, reward, terminated


class JacksCar(Env):
    '''
    Jack's car rental env
    '''

    def __init__(self) -> None:
        pass


class ShortCorridor(Env):
    '''
    Short corridor env
    '''

    def __init__(self, n_states: int,
                start_state: int,
                terminal_state: int, 
                switched_states: List[int]) -> None:
        '''
        Params
        ------
        n_states: number of states
        start_state: start state
        terminal_state: terminal state
        switched_states: list of switched-action state
        '''
        self.state_space = np.arange(n_states)
        self.action_space = [0, 1]
        self.start_state = start_state
        self.terminal_state = terminal_state
        self.switched_states = switched_states
        self.low = 0
        self.high = terminal_state


    def reset(self) -> int:
        '''
        Reset env
        '''
        self.state = self.start_state
        return self.state


    def _terminated(self) -> bool:
        '''
        Whether agent is in terminal state
        '''
        return self.state == self.terminal_state


    def _get_reward(self) -> float:
        '''
        Get reward

        Returns
        -------
        reward: reward corresponding
        '''
        return -1


    def step(self, action: int) -> Tuple[int, float, bool]:
        '''
        Take action

        Returns
        -------
        next_state: next state
        reward: corresponding reward
        terminated: whether the agent has reached the terminal state
        '''
        assert action in self.action_space, 'Invalid action!'
        assert self.state is not None, 'Call reset before using step method'

        action = 2 * action - 1
        if self.state in self.switched_states:
            action *= -1
        next_state = self.state + action
        next_state = max(min(next_state, self.high), self.low)

        self.state = next_state
        reward = self._get_reward()
        terminated = self._terminated()
        if terminated:
            reward = 0

        return next_state, reward, terminated


class BairdCounterexample(Env):
    '''
    Baird's counterexample
    '''

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states
        self.state_space = np.arange(n_states)

