import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


rewards = {'win': 1, 'draw': 0, 'lose': -1, 'ingame_r': 0}
actions = {'hits': 1, 'sticks': 0, 'busts': -1}


def get_card():
    card = np.random.randint(1, 14)
    return min(card, 10)


def get_card_value(card, player_usable_card):
    if card == 1 and player_usable_card:
        return 11
    return card


def is_usable(card, player_sum):
    return True if card == 1 and player_sum < 11 else False


def play():
    game_trajectory = []

    # initial player's card
    player_sum = 0
    player_usable_card = False

    player_card_1 = get_card()
    player_sum += get_card_value(player_card_1, True)

    player_card_2 = get_card()
    player_usable_card = is_usable(player_card_2, player_sum)
    player_sum += get_card_value(player_card_2, player_usable_card)

    # initial dealer's card
    dealer_sum = 0
    dealer_usable_card = False

    dealer_showed_card = get_card()
    dealer_hiden_card = get_card()

    dealer_sum += get_card_value(dealer_showed_card, True)
    dealer_usable_card = is_usable(dealer_hiden_card, dealer_sum)
    dealer_sum += get_card_value(dealer_hiden_card, dealer_usable_card)

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
                prev_state = game_trajectory[-1][0]
                prev_player_sum = prev_state[0]
                game_trajectory.append((prev_state, action, rewards['lose']))
                return game_trajectory
        elif player_sum == 20 or player_sum == 21:
            break
        else:
            action = actions['hits']
            game_trajectory.append((state, action, rewards['ingame_r']))

            card = get_card()
            player_usable_card = is_usable(card, player_sum)
            player_sum += get_card_value(card, player_usable_card)

    # dealer's turn (happens when player sticks)
    action = actions['sticks']
    while True:

        if dealer_sum > 21:
            if dealer_usable_card:
                dealer_sum -= 10
                dealer_usable_card = False
            else:
                game_trajectory.append((state, action, rewards['win']))
                return game_trajectory
        elif dealer_sum >= 17:
            break
        else:
            card = get_card()
            dealer_usable_card = is_usable(card, dealer_sum)
            dealer_sum += get_card_value(card, dealer_usable_card)

    # compare the final sum between player and dealer (happens when both sum <= 21)
    state = [player_sum, player_usable_card, dealer_showed_card]
    if player_sum > dealer_sum:
        game_trajectory.append((state, action, rewards['win']))
    elif player_sum == dealer_sum:
        game_trajectory.append((state, action, rewards['draw']))
    else:
        game_trajectory.append((state, action, rewards['lose']))

    return game_trajectory


def first_visit_MC(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for i in tqdm(range(episodes)):
        game_trajectory = play()

        for (state, action, reward) in game_trajectory:
            # since player's sum in range [10, 21]
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

    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


if __name__ == '__main__':
    states_usable_ace_1, states_no_usable_ace_1 = first_visit_MC(10000)
    states_usable_ace_2, states_no_usable_ace_2 = first_visit_MC(500000)

    states = [states_usable_ace_1, states_usable_ace_2, states_no_usable_ace_1, states_no_usable_ace_2]

    fig, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    axes = axes.flatten()

    titles = ['Usable ace, 10000 eps', 'Usable ace, 500000 eps', 'No usable ace, 10000 eps', 'No usable ace, 5000000 eps']

    for state, ax, title in zip(states, axes, titles):
        ax.pcolor(state, cmap=cm.coolwarm)
        ax.set_xlabel('Player sum', fontsize=8)
        ax.set_ylabel('Dealer showing', fontsize=8)
        ax.set_title(title, fontsize=9)
    plt.savefig('./blackjack_first_visit_MC.png')
    plt.close()
