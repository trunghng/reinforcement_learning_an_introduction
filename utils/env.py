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
    def step(self):
        pass


class RandomWalk(Env):
    '''
    Random walk env
    '''

    def __init__(self, n_states: int, start_state: int,
                terminal_states: List[int], 
                action_space: List[int]=[-1, 1],
                reward_space: List[float]=[1, 0, -1],
                transition_probs: Dict[int, float]={-1: 0.5, 1: 0.5}) -> None:
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
        self.reset()


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
        return self.state in self.terminal_states


    def _get_reward(self, next_state: int) -> float:
        '''
        Get reward corresponding to the next state @next_state

        Params
        ------
        next_state: next state of the agent

        Return
        ------
        reward: reward corresponding
        '''
        if next_state == self.terminal_states[1]:
            reward = self.reward_space[0]
        elif next_state == self.terminal_states[0]:
            reward = self.reward_space[-1]
        else:
            reward = self.reward_space[1]
        return reward


    def step(self, action: int, state: int=None) \
            -> Tuple[int, float, bool]:

        '''
        Take action

        Params
        ------
        action: action taken
        state: state of the agent

        Return
        ------
        next_state: next state
        reward: corresponding reward
        reward: whether next state is a terminal state
        '''
        assert action in self.action_space, "Invalid action!"
        if state is not None:
            self.state = state
        next_state = self.state + action
        self.state = next_state
        reward = self._get_reward(next_state)
        terminated = self._terminated()
        return next_state, reward, terminated


class TransitionRadiusRandomWalk(RandomWalk):
    '''
    Random walk with transition radius env
    '''

    def __init__(self, n_states: int, start_state: int,
                terminal_states: List[int], 
                action_space: List[int]=[-1, 1],
                reward_space: List[float]=[-1, 0, 1],
                transition_probs: Dict[int, float]={-1: 0.5, 1: 0.5},
                transition_radius: int=None) -> None:
        '''
        Params
        ------
        n_states: number of state
        start_state: start state
        terminal_states: list of terminal states
        action_space: action space
        reward_space: reward_space
        transition_probs: transition probabilities
        transition_radius: transition radius
        '''
        super().__init__(n_states, start_state, terminal_states, 
            action_space, reward_space, transition_probs)
        self.transition_radius = transition_radius


    def get_state_transition(self, state: int, action: int) -> Dict[int, float]:
        '''
        Get state transition at state @state

        Params
        ------
        state: state of the agent
        action: action taken at state @state

        Return
        ------
        state_transition: state transition probabilities
        '''
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


    def step(self, action: int, state: int=None) \
            -> Tuple[int, float, bool]:

        '''
        Take action

        Params
        ------
        action: action taken
        state: state of the agent

        Return
        ------
        next_state: next state
        reward: corresponding reward
        reward: whether next state is a terminal state
        '''
        assert action in self.action_space, "Invalid action!"
        if state is not None:
            self.state = state
        step = np.random.randint(1, self.transition_radius + 1)
        next_state = min(self.terminal_states[1], 
            max(self.terminal_states[0], self.state + action * step))
        self.state = next_state
        reward = self._get_reward(next_state)
        terminated = self._terminated()
        return next_state, reward, terminated


class GridWorld(Env):
    '''
    Gridworld env
    '''

    def __init__(self):
        pass


    def reset(self) -> None:
        pass


    def _terminated(self) -> bool:
        pass


    def step(self, action: int) -> Tuple[int, float, bool]:
        pass


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

        Return
        ------
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


    def step(self, action: Tuple[int, int]) \
            -> Tuple[np.ndarray, float, bool]:
        '''
        Take action

        Return
        ------
        next_state: next state
        reward: corresponding reward
        terminated: whether the car has finished the track
        '''
        assert action in self.action_space, "Invalid action!"

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

        reward = -1.0
        next_state = np.copy(self.state)

        return next_state, reward, terminated
