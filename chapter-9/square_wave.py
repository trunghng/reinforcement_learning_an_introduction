import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class Interval:
    '''
    One-dimensional receptive fields are overlapping intervals rather than circles
    '''

    def __init__(self, left_end_pt, right_end_pt):
        self.left_end_pt = left_end_pt
        self.right_end_pt = right_end_pt


    def is_inside(self, x):
        '''
        Whether @x is inside the interval

        Params
        ------
        @x: one-dimensional point
        '''
        return self.left_end_pt <= x <= self.right_end_pt


def square_wave(x, interval):
    '''
    Square wave function

    Params
    ------
    x: float
        point
    interval: Interval
    '''
    if interval.is_inside(x):
        return 1
    return 0



if __name__ == '__main__':
    n_examples = [10, 40, 160, 640, 2560, 10240]
    n_features = 50
    step_size = 0.2 / n_features


