from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np


class Env(ABC):


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
                trans_probs: Dict[int, float]={-1: 0.5, 1: 0.5}) -> None:
        '''
        Params
        ------
        n_states: number of state
        start_state: start state
        terminal_states: list of terminal states
        actions: action space
        trans_probs: transition probabilities
        '''
        self.n_states = n_states
        self.state_space = np.arange(1, n_states + 1)
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.action_space = action_space
        self.reward_space = reward_space
        self.trans_probs = trans_probs
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


    def get_reward(self, next_state: int) -> float:
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
        next_state = self.state + action
        self.state = next_state
        reward = self.get_reward(next_state)
        terminated = self._terminated()
        return next_state, reward, terminated

