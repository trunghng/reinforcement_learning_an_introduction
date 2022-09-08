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
    def reset(self) -> None:
        pass


    @abstractmethod
    def _terminated(self) -> bool:
        pass


    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool]:
        pass


class RandomWalk(Env):
    '''
    Random walk env
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
        actions: action space
        transition_probs: transition probabilities
        transition_radius: transition radius
        '''
        self.n_states = n_states
        self.state_space = np.arange(1, n_states + 1)
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.action_space = action_space
        self.reward_space = reward_space
        self.transition_probs = transition_probs
        self.transition_radius = transition_radius
        self.reset()


    def reset(self) -> None:
        '''
        Reset env
        '''
        self.state = self.start_state


    def _terminated(self) -> bool:
        '''
        Whether agent is in terminal state

        Return
        ------
        terminated: whether the state of the agent
            is a terminal state
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
        if next_state == self.terminal_states[0]:
            reward = self.reward_space[0]
        elif next_state == self.terminal_states[1]:
            reward = self.reward_space[2]
        else:
            reward = self.reward_space[1]
        return reward


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
        (next_state, reward, terminated): tuple of next state, 
            reward corresponding, and is terminated
        '''
        assert action in self.action_space, "Invalid action!"
        if state is not None:
            self.state = state
        if self.transition_radius is None:
            next_state = self.state + action
        else:
            step = np.random.randint(1, self.transition_radius + 1)
            next_state = min(self.terminal_states[1], 
                max(self.terminal_states[0], self.state + action * step))
        self.state = next_state
        reward = self._get_reward(next_state)
        terminated = self._terminated()
        return next_state, reward, terminated

