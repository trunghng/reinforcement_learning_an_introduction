import numpy as np
import matplotlib.pyplot as plt


class Interval:
    '''
    One-dimensional receptive fields are overlapping intervals rather than circles
    '''

    def __init__(self, left_end_pt, right_end_pt):
        self.left_end_pt = left_end_pt
        self.right_end_pt = right_end_pt


    def contains(self, x):
        '''
        Whether @x is inside the interval

        Params
        ------
        x: float
        '''
        return self.left_end_pt <= x <= self.right_end_pt


    def get_length(self):
        '''
        Get the length of the interval
        '''
        return self.right_end_pt - self.left_end_pt


class ValueFunction:

    def __init__(self, n_features, feature_width, interval):
        self.w = np.zeros(n_features)
        self.n_features = n_features
        self.feature_width = feature_width
        self.features = self._get_features(interval)


    def _get_features(self, interval):
        '''
        Divide the interval into @self.n_features intervals (each interval corresponds to a feature)
        such that each interval has length of @self.feature_width

        Params
        ------
        interval: Interval
        '''
        features = []
        step = (interval.get_length() - self.feature_width) / (self.n_features - 1)
        sub_interval_left_end_pt = interval.left_end_pt

        for _ in range(n_features - 1):
            sub_interval = Interval(sub_interval_left_end_pt, sub_interval_left_end_pt + self.feature_width)
            features.append(sub_interval)
            sub_interval_left_end_pt += step
        features.append(Interval(sub_interval_left_end_pt, interval.right_end_pt))

        return features


    def get_feature_vector(self, x):
        '''
        Get the feature vector corresponding to @x:
        a feature containing @x gives value of 1, and 0 otherwise 

        Params
        ------
        x: float
        '''
        feature_vector = np.zeros(self.n_features)

        for i in range(self.n_features):
            if self.features[i].contains(x):
                feature_vector[i] = 1

        return feature_vector


    def get_grad(self, x):
        '''
        Compute the gradient w.r.t @self.w at @x:
        since value function is approximated by a linear function,
        its gradient w.r.t weight @self.w is equal to the feature vector

        Params
        ------
        x: float
        '''
        feature_vector = self.get_feature_vector(x)
        grad = feature_vector

        return grad


    def get_value(self, x):
        '''
        Get the value function at @x

        Params
        ------
        x: float
        '''
        feature_vector = self.get_feature_vector(x)
        value_function = np.dot(self.w, feature_vector)

        return value_function


def square_wave(interval, x):
    '''
    Square wave function, which return 1 if @x is inside
    the @interval and 0 otherwise

    Params
    ------
    interval: Interval
    x: float
    '''
    if interval.contains(x):
        return 1
    return 0


def sample(n_samples, interval):
    '''
    Generate @n_samples points uniformly inside @interval

    Params
    ------
    n_sample: int
        number of samples
    interval: Interval
    '''
    samples = np.random.uniform(interval.left_end_pt, interval.right_end_pt, n_samples)
    return samples


if __name__ == '__main__':
    n_features = 50
    step_size = 0.2 / n_features
    feature_widths = [0.2, 0.4, 1]
    n_samples_list = [10, 40, 160, 640, 2560, 10240]
    domain = Interval(0, 2)
    interval = Interval(0.5, 1.5)

    plt.figure(figsize=(21, 14))
    axis_x = np.arange(domain.left_end_pt, domain.right_end_pt, 0.02)
    for index, n_samples in enumerate(n_samples_list):
        print(n_samples, 'samples')
        samples = sample(n_samples, domain)
        values = []
        for x in samples:
            values.append(square_wave(interval, x))

        value_functions = [ValueFunction(n_features, feature_width, domain) 
                            for feature_width in feature_widths]
        plt.subplot(2, 3, index + 1)
        plt.title('%d samples' % (n_samples))
        for value_function in value_functions:
            for x, value in zip(samples, values):
                value_function.w += step_size * (value - value_function.get_value(x)) \
                                    * value_function.get_grad(x)

            values_ = [value_function.get_value(x_) for x_ in axis_x]
            plt.plot(axis_x, values_, label='feature width %.01f' % (value_function.feature_width))
        plt.legend(loc='upper right')

    plt.savefig('./square_wave_function.png')
    plt.close()
