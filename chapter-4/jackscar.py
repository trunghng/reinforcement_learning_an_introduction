import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import poisson

MAX_CARS = 20
MAX_CARS_MOVE = 5
GAMMA = 0.9  # discount factor
EXPECTED_REQUEST = [3, 4]
EXPECTED_RETURN = [3, 2]
RENTAL_CREDIT = 10
MOVING_COST = 2
POISSON_UPPER_BOUND = 11
actions = np.arange(-MAX_CARS_MOVE, MAX_CARS_MOVE + 1)
poisson_dict = dict()


def poisson_probability(n, poisson_lambda):
    global poisson_dict
    key = n * 10 + poisson_lambda
    if key not in poisson_dict:
        poisson_dict[key] = poisson.pmf(n, poisson_lambda)

    return poisson_dict[key]


def expected_return(state, action, state_value):
    exp_return = 0
    # total moving cost
    # exp_return -= abs(action) * MOVING_COST
    NO_CARS_FIRST_LOC = min(MAX_CARS, state[0] - action)
    NO_CARS_SECOND_LOC = min(MAX_CARS, state[1] + action)

    for no_requests_first_loc in range(POISSON_UPPER_BOUND):
        for no_requests_second_loc in range(POISSON_UPPER_BOUND):
            no_cars_first_loc = NO_CARS_FIRST_LOC
            no_cars_second_loc = NO_CARS_SECOND_LOC

            request_prob = poisson_probability(no_requests_first_loc, EXPECTED_REQUEST[
                                               0]) * poisson_probability(no_requests_second_loc, EXPECTED_REQUEST[1])

            no_valid_requests_first_loc = min(no_cars_first_loc, no_requests_first_loc)
            no_valid_requests_second_loc = min(no_cars_second_loc, no_requests_second_loc)

            reward = (no_valid_requests_first_loc + no_valid_requests_second_loc) * RENTAL_CREDIT
            reward -= abs(action) * MOVING_COST
            no_cars_first_loc -= no_valid_requests_first_loc
            no_cars_second_loc -= no_valid_requests_second_loc

            # Uncomment this if the number of cars returned is a constant
            # """
            for no_returns_first_loc in range(POISSON_UPPER_BOUND):
                for no_returns_second_loc in range(POISSON_UPPER_BOUND):

                    return_prob = poisson_probability(no_returns_first_loc, EXPECTED_RETURN[
                                                      0]) * poisson_probability(no_returns_second_loc, EXPECTED_RETURN[1])
                    no_cars_first_loc = min(MAX_CARS, no_cars_first_loc + no_returns_first_loc)
                    no_cars_second_loc = min(MAX_CARS, no_cars_second_loc + no_returns_second_loc)
                    exp_return += request_prob * return_prob * (reward + GAMMA * state_value[no_cars_first_loc, no_cars_second_loc])
            # """

            # comment this if the number of cars returned is a Poisson rv
            """
            no_returns_first_loc = EXPECTED_RETURN[0]
            no_returns_second_loc = EXPECTED_RETURN[1]
            no_cars_first_loc = min(MAX_CARS, no_cars_first_loc + no_returns_first_loc)
            no_cars_second_loc = min(MAX_CARS, no_cars_second_loc + no_returns_second_loc)
            exp_return += request_prob * (reward + GAMMA * state_value[no_cars_first_loc, no_cars_second_loc])
            """

    return exp_return


def policy_iteration(theta):
    state_value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), np.int8)

    iter_count = 0
    fig = plt.figure(figsize=(10.5, 7))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    # axes = axes.flatten()
    while True:
        if iter_count < 5:
            # Policy Evaluation
            ax = fig.add_subplot(2, 3, iter_count + 1)
            ax.pcolor(policy, cmap=cm.coolwarm)
            ax.set_title('Policy {}'.format(iter_count), fontsize=9)
            ax.set_xlabel('#Cars at second location', fontsize=8)
            ax.set_ylabel('#Cars at first location', fontsize=8)

        while True:
            old_value = state_value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    state_value[i, j] = expected_return([i, j], policy[i, j], state_value)
            delta = np.max(abs(old_value - state_value))
            print('max value change {}'.format(delta))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_values = []
                for action in actions:
                    if 0 <= action <= i or -j <= action <= 0:
                        action_values.append(expected_return([i, j], action, state_value))
                    else:
                        action_values.append(-np.inf)
                policy[i, j] = actions[np.argmax(action_values)]
                if policy_stable and old_action != policy[i, j]:
                    policy_stable = False
        if policy_stable:
            ax = fig.add_subplot(2, 3, 6, projection='3d')
            X = np.arange(MAX_CARS + 1)
            Y = np.arange(MAX_CARS + 1)
            X, Y = np.meshgrid(X, Y)
            Z = state_value
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            ax.set_title('Optimal value', fontsize=9)
            ax.set_xlabel('#Cars at second location', fontsize=8)
            ax.set_ylabel('#Cars at first location', fontsize=8)

            plt.show()
            break

        iter_count += 1

    return state_value, policy


if __name__ == '__main__':
    theta = 1e-4
    optimal_state_value, optimal_policy = policy_iteration(theta)
