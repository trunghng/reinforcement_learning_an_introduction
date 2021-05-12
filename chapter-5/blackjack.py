import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


rewards = {'win': 1, 'draw': 0, 'lose': -1}
actions = {'hits': 1, 'sticks': 0, 'busts': -1}


def get_card():
    card = np.random.randint(1, 14)
    return min(card, 10)


def get_card_value(card, player_usable_card):
    if card == 1 and player_usable_card:
        return 11
    return card


def is_usable(card, player_sum):
    return card == 1 and player_sum < 11


def play():
    game_trajectory = []

    # initial player's card
    player_sum = 0
    player_usable_card = False

    while player_sum < 12:
        card = get_card()
        if not player_usable_card:
            player_usable_card = is_usable(card, player_sum)
        player_sum += get_card_value(card, is_usable(card, player_sum))

    # initial dealer's card
    dealer_sum = 0
    dealer_usable_card = False

    dealer_showed_card = get_card()
    dealer_hiden_card = get_card()

    dealer_usable_card = is_usable(dealer_showed_card, dealer_sum)
    dealer_sum += get_card_value(dealer_showed_card, dealer_usable_card)
    if not dealer_usable_card:
        dealer_usable_card = is_usable(dealer_hiden_card, dealer_sum)
    dealer_sum += get_card_value(dealer_hiden_card, is_usable(dealer_hiden_card, dealer_sum))

    # game starts
    # player's turn
    # use the strategy that to keep hitting while the card sum < 20
    state = []
    while True:
        state = [player_sum, player_usable_card, dealer_showed_card]

        if player_sum > 21:
            if player_usable_card:
                player_sum -= 10
                player_usable_card = False
            else:
                action = actions['busts']
                return (game_trajectory, rewards['lose'])
        elif player_sum == 20 or player_sum == 21:
            break
        else:
            action = actions['hits']
            game_trajectory.append((state, action))

            card = get_card()
            if not player_usable_card:
                player_usable_card = is_usable(card, player_sum)
            player_sum += get_card_value(card, is_usable(card, player_sum))

    # dealer's turn (happens when player sticks)
    action = actions['sticks']
    while True:
        if dealer_sum > 21:
            if dealer_usable_card:
                dealer_sum -= 10
                dealer_usable_card = False
            else:
                game_trajectory.append((state, action))
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
        game_trajectory.append((state, action))
        return (game_trajectory, rewards['win'])
    elif player_sum == dealer_sum:
        game_trajectory.append((state, action))
        return (game_trajectory, rewards['draw'])
    else:
        game_trajectory.append((state, action))
        return (game_trajectory, rewards['lose'])


# first-visit Monte Carlo prediction
def first_visit_MC(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.zeros((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.zeros((10, 10))

    for i in tqdm(range(episodes)):
        game_trajectory, reward = play()
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

    for i in range(states_usable_ace_count.shape[0]):
        for j in range(states_usable_ace_count.shape[1]):
            if states_usable_ace_count[i, j] == 0:
                states_usable_ace_count[i, j] += 1
            if states_no_usable_ace_count[i, j] == 0:
                states_no_usable_ace_count[i, j] += 1

    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


def on_policy_first_visit_MC(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for i in tqdm(range(episodes)):
        game_trajectory, reward = play()

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


if __name__ == '__main__':
    states_usable_ace_1, states_no_usable_ace_1 = first_visit_MC(10000)
    states_usable_ace_2, states_no_usable_ace_2 = first_visit_MC(500000)

    states = [states_usable_ace_1, states_usable_ace_2, states_no_usable_ace_1, states_no_usable_ace_2]

    fig, axes = plt.subplots(2, 2, figsize=(30, 22.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    axes = axes.flatten()

    titles = ['Usable ace, 10000 eps', 'Usable ace, 500000 eps', 'No usable ace, 10000 eps', 'No usable ace, 5000000 eps']

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
