import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


rewards = {'win': 1, 'draw': 0, 'lose': -1}
actions = {'hits': 0, 'sticks': 1}
np.random.seed(27)


def get_card():
    card = np.random.randint(1, 14)
    return min(card, 10)


def get_card_value(card, player_usable_card):
    if card == 1 and player_usable_card:
        return 11
    return card


def is_usable(card, player_sum):
    return card == 1 and player_sum < 11


# the strategy that to keep hitting while the cards' sum < 20
def player_policy(player_sum, player_usable_card, dealer_showed_card):
    if player_sum == 20 or player_sum == 21:
        return actions['sticks']
    else:
        return actions['hits']


def play(player_policy, initial_state=None, initial_action=None):
    game_trajectory = []

    # initial player's card and dealer's card
    player_sum = 0
    player_usable_card = False
    dealer_showed_card = 0
    dealer_hiden_card = 0

    if initial_state is None:
        while player_sum < 12:
            card = get_card()
            if not player_usable_card:
                player_usable_card = is_usable(card, player_sum)
            player_sum += get_card_value(card, is_usable(card, player_sum))

        dealer_showed_card = get_card()
        dealer_hiden_card = get_card()
    else:
        player_sum, player_usable_card, dealer_showed_card = initial_state
        dealer_hiden_card = get_card()

    dealer_sum = 0
    dealer_usable_card = False

    dealer_usable_card = is_usable(dealer_showed_card, dealer_sum)
    dealer_sum += get_card_value(dealer_showed_card, dealer_usable_card)
    if not dealer_usable_card:
        dealer_usable_card = is_usable(dealer_hiden_card, dealer_sum)
    dealer_sum += get_card_value(dealer_hiden_card, is_usable(dealer_hiden_card, dealer_sum))

    # game starts
    # player's turn
    state = []
    while True:
        state = [player_sum, player_usable_card, dealer_showed_card]

        if player_sum > 21:
            if player_usable_card:
                player_sum -= 10
                player_usable_card = False
            else:
                # player goes bust
                return (game_trajectory, rewards['lose'])
        else:
            if initial_action is not None:
                action = initial_action
                initial_action = None
            else:
                action = player_policy(player_sum, player_usable_card, dealer_showed_card)

            game_trajectory.append((state, action))
            if action == actions['sticks']:
                break

            card = get_card()
            if not player_usable_card:
                player_usable_card = is_usable(card, player_sum)
            player_sum += get_card_value(card, is_usable(card, player_sum))
            

    # dealer's turn (happens when player sticks)
    while True:
        if dealer_sum > 21:
            if dealer_usable_card:
                dealer_sum -= 10
                dealer_usable_card = False
            else:
                return (game_trajectory, rewards['win'])
        elif dealer_sum >= 17:
            break
        else:
            card = get_card()
            if not dealer_usable_card:
                dealer_usable_card = is_usable(card, dealer_sum)
            dealer_sum += get_card_value(card, is_usable(card, dealer_sum))

    # compare the final sum between player and dealer (happens when both sum <= 21)
    state = [player_sum, player_usable_card, dealer_showed_card]
    if player_sum > dealer_sum:
        return (game_trajectory, rewards['win'])
    elif player_sum == dealer_sum:
        return (game_trajectory, rewards['draw'])
    else:
        return (game_trajectory, rewards['lose'])


# first-visit Monte Carlo prediction
def first_visit_MC(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.zeros((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.zeros((10, 10))

    for _ in tqdm(range(episodes)):
        game_trajectory, reward = play(player_policy)
        visited_usable_ace = np.full((10, 10), False)
        visited_no_usable_ace = np.full((10, 10), False)

        for (state, _) in game_trajectory:
            # since player's sum in range [12, 21]
            player_sum = state[0] - 12
            # since dealer's card in range [1, 10]
            dealer_card = state[2] - 1

            if state[1]:
                if not visited_usable_ace[player_sum, dealer_card]:
                    states_usable_ace[player_sum, dealer_card] += reward
                    states_usable_ace_count[player_sum, dealer_card] += 1
                    visited_usable_ace[player_sum, dealer_card] = True
            else:
                if not visited_no_usable_ace[player_sum, dealer_card]:
                    states_no_usable_ace[player_sum, dealer_card] += reward
                    states_no_usable_ace_count[player_sum, dealer_card] += 1
                    visited_no_usable_ace[player_sum, dealer_card] = True

    states_usable_ace_count[states_usable_ace_count == 0] = 1
    states_no_usable_ace_count[states_no_usable_ace_count == 0] = 1

    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


def on_policy_first_visit_MC(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for _ in tqdm(range(episodes)):
        game_trajectory, reward = play(player_policy)

        for (state, _) in game_trajectory:
            # since player's sum in range [12, 21]
            player_sum = state[0] - 12
            # since dealer's card in range [1, 10]
            dealer_card = state[2] - 1

            # If player having usable card
            if state[1]:
                states_usable_ace[player_sum, dealer_card] += reward
                states_usable_ace_count[player_sum, dealer_card] += 1
            else:
                states_no_usable_ace[player_sum, dealer_card] += reward
                states_no_usable_ace_count[player_sum, dealer_card] += 1

    # for i in range(states_usable_ace_count.shape[0]):
    #     for j in range(states_usable_ace_count.shape[1]):
    #         if states_usable_ace_count[i, j] == 0:
    #             states_usable_ace_count[i, j] += 1
    #         if states_no_usable_ace_count[i, j] == 0:
    #             states_no_usable_ace_count[i, j] += 1

    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


# Monte Carlo Exploring Starts
def monte_carlo_ES(episodes):
    # state = [player_sum, player_usable_card, dealer_showed_card]
    state_action_values = np.zeros((10, 2, 10, 2))
    state_action_pair_count = np.zeros((10, 2, 10, 2))

    def greedy_policy(player_sum, player_usable_card, dealer_showed_card):
        player_sum -= 12
        player_usable_card = int(player_usable_card)
        dealer_showed_card -= 1

        values = state_action_values[player_sum, player_usable_card, dealer_showed_card, :] /\
                    np.where(state_action_pair_count[player_sum, player_usable_card, dealer_showed_card, :] == 0,
                        state_action_pair_count[player_sum, player_usable_card, dealer_showed_card, :], 1)
        return np.random.choice(np.argwhere(values == np.amax(values)).flatten().tolist())

    for eps in tqdm(range(episodes)):
        initial_state = [np.random.choice(range(12, 22)), np.random.choice([True, False]), np.random.choice(range(1, 11))]
        initial_action = np.random.choice([actions['hits'], actions['sticks']])
        policy = greedy_policy if eps else player_policy

        game_trajectory, reward = play(policy, initial_state, initial_action)
        first_visit_set = set()

        for state, action in game_trajectory:
            player_sum = state[0] - 12
            player_usable_card = int(state[1])
            dealer_showed_card = state[2] - 1
            state_action = (player_sum, player_usable_card, dealer_showed_card, action)
            if state_action not in first_visit_set:
                first_visit_set.add(state_action)
                state_action_values[player_sum, player_usable_card, dealer_showed_card, action] += reward
                state_action_pair_count += 1

    state_action_pair_count[state_action_pair_count == 0] = 1
    return state_action_values / state_action_pair_count



def first_visit_MC_plot():
    states_usable_ace_1, states_no_usable_ace_1 = first_visit_MC(10000)
    states_usable_ace_2, states_no_usable_ace_2 = first_visit_MC(500000)

    states = [states_usable_ace_1, states_usable_ace_2, states_no_usable_ace_1, states_no_usable_ace_2]
    titles = ['Usable ace, 10000 eps', 'Usable ace, 500000 eps', 'No usable ace, 10000 eps', 'No usable ace, 5000000 eps']

    fig, axes = plt.subplots(2, 2, figsize=(30, 22.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    axes = axes.flatten()

    for state, ax, title in zip(states, axes, titles):
        pc = ax.pcolor(state, cmap=cm.coolwarm)
        ax.set_title(title, fontsize=27)

        ax.set_ylabel('Player sum', fontsize=20)
        y_ticks_loc = range(12, 22)
        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.arange(y_start, y_end) + 0.5, minor=False)
        ax.set_yticklabels(y_ticks_loc)

        ax.set_xlabel('Dealer showing', fontsize=20)
        x_ticks_loc = range(1, 11)
        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.arange(x_start, x_end) + 0.5, minor=False)
        ax.set_xticklabels(x_ticks_loc)

        fig.colorbar(pc, ax=ax)

    plt.savefig('./blackjack_first_visit_MC.png')
    plt.close()


def monte_carlo_ES_plot():
    state_action_values = monte_carlo_ES(500000)

    # optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, 0, :, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, 1, :, :], axis=-1)

    # optimal state-value function
    state_value_no_usable_ace = np.max(state_action_values[:, 0, :, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, 1, :, :], axis=-1)

    plots = [action_usable_ace, state_value_usable_ace, action_no_usable_ace, state_value_no_usable_ace]
    titles = ['Optimal policy, usable ace', 'Optimal state-value function, usable ace', 
                'Optimal policy, no usable ace', 'Optimal state-value function, no usable ace']

    fig, axes = plt.subplots(2, 2, figsize=(30, 22.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    axes = axes.flatten()

    for plot, ax, title in zip(plots, axes, titles):
        pc = ax.pcolor(plot, cmap=cm.coolwarm)
        ax.set_title(title, fontsize=27)

        ax.set_ylabel('Player sum', fontsize=20)
        y_ticks_loc = range(12, 22)
        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.arange(y_start, y_end) + 0.5, minor=False)
        ax.set_yticklabels(y_ticks_loc)

        ax.set_xlabel('Dealer showing', fontsize=20)
        x_ticks_loc = range(1, 11)
        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.arange(x_start, x_end) + 0.5, minor=False)
        ax.set_xticklabels(x_ticks_loc)

        fig.colorbar(pc, ax=ax)

    plt.savefig('./blackjack_monte_carlo_es.png')
    plt.close()


if __name__ == '__main__':
    first_visit_MC_plot()
    # monte_carlo_ES_plot()

