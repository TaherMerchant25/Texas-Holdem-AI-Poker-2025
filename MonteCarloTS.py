import random
from itertools import combinations
import time


class HandEval:
    def __init__(self):
        self.SUITES = ("d", "c", "h", "s")
        self.CARD_VALUES = {
            "J": 11,
            "D": 12,
            "K": 13,
            "A": 14,
            **{str(n): n for n in range(2, 11)},
        }
        self.deck = []
        self.reset_deck()
        self.table = set()
        self.hand = set()

    def set_table_as(self, table, normalize=True):
        if normalize:
            self.table = set(self._normalize_card(card) for card in table)
        else:
            self.table = set(table)
        self.deck -= self.table

    def update_table(self, card, normalize=True):
        if normalize:
            card = self._normalize_card(card)
        self.table.add(card)
        self.deck -= {card}

    def update_hand(self, players_hand, normalize=True):
        if normalize:
            self.hand = set(self._normalize_card(card) for card in players_hand)
        else:
            self.hand = set(player_hand)
        self.deck -= self.hand

    def reset_deck(self):
        self.deck = set(
            self._normalize_card((num, col))
            for num in self.CARD_VALUES
            for col in self.SUITES
        )

    def _normalize_card(self, card):
        number = self.CARD_VALUES[card[0]]
        color = self.SUITES.index(card[1])
        return number, color

    def positions_to_calculate(self, num_players=2):
        """
        Calculates the total number of possible positions to evaluate.
        :param num_players: Total number of players (including the user).
        :return: Total number of positions.
        """
        n_cards_left_in_deck = len(self.deck)
        n_cards_left_to_full_table = 5 - len(self.table)
        multiplier = 1
        denominator = 1

        # Calculate combinations for completing the table
        for i in range(n_cards_left_to_full_table):
            multiplier *= n_cards_left_in_deck - i
            denominator *= i + 1

        # Calculate combinations for distributing hands to all players
        cards_to_choose_from_for_players = (
                n_cards_left_in_deck - n_cards_left_to_full_table
        )
        for i in range(2 * (num_players - 1)):  # Exclude the user's hand
            multiplier *= cards_to_choose_from_for_players - i
            denominator *= i + 1

        return multiplier // denominator

    def value_of_enemy_cards_is_smaller(self, full_table, enemy_cards, our_cards_value):
        cards7_enemy = tuple(sorted(full_table + enemy_cards))
        enemy_cards_value = self.value_of_best_hand_type_from_seven_cards(cards7_enemy)
        return our_cards_value >= enemy_cards_value

    def calculate_position(self, num_players=2):
        """
        Calculates the exact win percentage for the player's hand against multiple players.
        :param num_players: Total number of players (including the user).
        :return: Win percentage.
        """
        win = 0
        for table_addition in combinations(self.deck, 5 - len(self.table)):
            full_table = tuple(self.table) + table_addition
            rest_of_deck = tuple(self.deck - set(table_addition))

            our_seven_cards = tuple(sorted(full_table + tuple(self.hand)))
            our_cards_value = self.value_of_best_hand_type_from_seven_cards(
                our_seven_cards
            )

            for enemy_hands in combinations(rest_of_deck, 2 * (num_players - 1)):
                enemy_hands = [
                    enemy_hands[i: i + 2] for i in range(0, len(enemy_hands), 2)
                ]
                enemy_values = [
                    self.value_of_best_hand_type_from_seven_cards(
                        tuple(sorted(full_table + tuple(enemy_hand)))
                    )
                    for enemy_hand in enemy_hands
                ]

                if all(our_cards_value >= enemy_value for enemy_value in enemy_values):
                    win += 1

        return round(win * 100 / self.positions_to_calculate(num_players), 2)

    def calculate_pos_monte_carlo(self, num_players=2, samples=2000):
        """
        Estimates the win percentage for the player's hand against multiple players using Monte Carlo sampling.
        :param num_players: Total number of players (including the user).
        :param samples: Number of random samples to take.
        :return: Estimated win percentage.
        """
        exact_simulations = self.positions_to_calculate(num_players)
        if exact_simulations <= samples:
            return self.calculate_position(num_players)

        win = 0
        deck_list = sorted(self.deck)  # Convert deck to a sorted list
        for _ in range(samples):
            # Randomly sample cards to complete the table
            table_addition = random.sample(deck_list, 5 - len(self.table))
            full_table = tuple(self.table) + tuple(table_addition)
            rest_of_deck = tuple(set(deck_list) - set(table_addition))

            # Calculate our hand value
            our_seven_cards = tuple(sorted(full_table + tuple(self.hand)))
            our_cards_value = self.value_of_best_hand_type_from_seven_cards(
                our_seven_cards
            )

            # Randomly sample enemy hands
            enemy_hands = [
                random.sample(rest_of_deck, 2) for _ in range(num_players - 1)
            ]
            enemy_values = [
                self.value_of_best_hand_type_from_seven_cards(
                    tuple(sorted(full_table + tuple(enemy_hand)))
                )
                for enemy_hand in enemy_hands
            ]

            if all(our_cards_value >= enemy_value for enemy_value in enemy_values):
                win += 1

        return round(win * 100 / samples, 2)

    def there_is_con_len5_in_cards7(self, card_numbers):
        for i in range(3):
            cards5 = card_numbers[2 - i: 7 - i]
            first_of_selection = cards5[0]
            if cards5 == list(range(first_of_selection, first_of_selection + 5)):
                return first_of_selection, i
        return 0, 0

    def check_if_same_suite_in_cards(self, cards7):
        colors = tuple(col for _, col in cards7)
        for color in self.SUITES:
            count = colors.count(color)
            if count >= 5:
                for num, col in cards7[::-1]:
                    if col == color:
                        return num
        return 0

    def value_of_best_hand_type_from_seven_cards(self, cards7):
        l_of_nums = tuple(num for num, _ in cards7)

        if 14 in l_of_nums[::-1]:
            l_of_nums = (0,) + l_of_nums

        high_card = l_of_nums[-1]

        (
            pair,
            two_pair,
            three_of_a_kind,
            straight,
            flush,
            full_house,
            four_of_a_kind,
            straight_flush,
        ) = range(1, 9)
        hand_types_on_table = [False] * 8
        hand_types_on_table[0] = True
        value_of_type = [0] * 8

        def update_hand_type_on_table(hand_type_index, value_of_hand_type):
            nonlocal hand_types_on_table, value_of_type
            hand_types_on_table[hand_type_index] = True
            value_of_type[hand_type_index] = value_of_hand_type

        card_starting_con5, index_start = self.there_is_con_len5_in_cards7(l_of_nums)
        if card_starting_con5 != 0:
            max_of_con5 = card_starting_con5 + 4
            if all(
                    cards7[2 - index_start][1] == col
                    for num, col in cards7[2 - index_start: 7 - index_start]
            ):
                return straight_flush * 10000 + max_of_con5 * 100
            else:
                update_hand_type_on_table(straight, max_of_con5)

        for number in l_of_nums:
            count = l_of_nums.count(number)

            if count == 4:
                return four_of_a_kind * 10000 + number * 100 + high_card

            elif count == 3:
                if hand_types_on_table[pair]:
                    update_hand_type_on_table(
                        full_house, number * 3 + value_of_type[pair] * 2
                    )
                elif hand_types_on_table[three_of_a_kind]:
                    other_three = value_of_type[three_of_a_kind]
                    update_hand_type_on_table(
                        full_house,
                        max(other_three, number) * 3 + min(other_three, number) * 2,
                    )
                else:
                    update_hand_type_on_table(three_of_a_kind, number)

            elif count == 2:
                if hand_types_on_table[two_pair]:
                    first_pair = value_of_type[pair]
                    both_pairs = value_of_type[two_pair]
                    second_pair = both_pairs - first_pair
                    a = number + first_pair
                    b = number + second_pair
                    update_hand_type_on_table(two_pair, max(a, b, both_pairs))
                elif hand_types_on_table[full_house]:
                    other_two = value_of_type[pair]
                    new_value = (
                            value_of_type[full_house]
                            + (max(number, other_two) - other_two) * 2
                    )
                    update_hand_type_on_table(full_house, new_value)
                elif hand_types_on_table[pair]:
                    update_hand_type_on_table(two_pair, value_of_type[pair] + number)
                elif hand_types_on_table[three_of_a_kind]:
                    update_hand_type_on_table(
                        full_house, value_of_type[three_of_a_kind] * 3 + number * 2
                    )
                else:
                    update_hand_type_on_table(pair, number)

        highest_card_in_flush = self.check_if_same_suite_in_cards(cards7)
        if highest_card_in_flush != 0:
            update_hand_type_on_table(flush, highest_card_in_flush)

        for i in range(8):
            current_index = 7 - i
            if hand_types_on_table[current_index]:
                count_highest_card = current_index not in {
                    straight,
                    flush,
                    full_house,
                    straight_flush,
                }
                return (
                        10000 * current_index
                        + 100 * value_of_type[current_index]
                        + high_card * count_highest_card
                )


class PokerHandEvaluator:
    def __init__(self):
        self.evaluator = HandEval()

    def normalize_cards_from_engine(self, card_indices):
        """
        Converts card indices from the poker engine into the normalized format used by the evaluator.
        :param card_indices: List of card indices from the poker engine.
        :return: List of normalized cards in the evaluator's format.
        """
        normalized = []
        for card in card_indices:
            if card > 0:  # Ignore cards with index 0 (not dealt)
                rank = (
                    str((card - 1) % 13 + 2)
                    if (card - 1) % 13 + 2 <= 10
                    else ["J", "Q", "K", "A"][(card - 1) % 13 - 10]
                )
                suit = ["s", "h", "d", "c"][(card - 1) // 13]
                normalized.append((rank, suit))
        return normalized

    def set_table(self, community_cards):
        """
        Sets the community cards for evaluation.
        :param community_cards: List of Card objects from the poker engine.
        """
        normalized_table = self.normalize_cards_from_engine(community_cards)
        self.evaluator.set_table_as(normalized_table)

    def set_hand(self, player_hand):
        """
        Sets the player's hand for evaluation.
        :param player_hand: List of Card objects from the poker engine.
        """
        normalized_hand = self.normalize_cards_from_engine(player_hand)
        self.evaluator.update_hand(normalized_hand)

    def calculate_exact(self, num_players):
        """
        Calculates the exact win percentage for the player's hand.
        :param num_players: Total number of players (including the user).
        :return: Exact win percentage.
        """
        return self.evaluator.calculate_position(num_players)

    def calculate_monte_carlo(self, num_players, samples=2000):
        """
        Estimates the win percentage using Monte Carlo sampling.
        :param num_players: Total number of players (including the user).
        :param samples: Number of random samples to take.
        :return: Estimated win percentage.
        """
        return self.evaluator.calculate_pos_monte_carlo(num_players, samples)

    def reset(self):
        self.evaluator.reset_deck()


if __name__ == "__main__":

    evaluator = HandEval()
    total_difference = 0
    total_time_difference = 0

    for i in range(20):
        evaluator.reset_deck()

        num_community_cards = random.randint(4, 5)
        community_cards = random.sample(sorted(evaluator.deck), num_community_cards)
        evaluator.set_table_as(community_cards, False)

        player_hand = random.sample(sorted(evaluator.deck), 2)
        evaluator.update_hand(player_hand, False)

        num_players = random.randint(2, 4)

        print(f"Test Case {i + 1}:")
        print(f"Community Cards: {community_cards}")
        print(f"Player Hand: {player_hand}")
        print(f"Number of Players: {num_players}")

        start_time = time.time()
        exact_result = evaluator.calculate_position(num_players)
        exact_time = time.time() - start_time
        print(f"  Exact calculation result: {exact_result}% (Time: {exact_time:.2f}s)")

        start_time = time.time()
        monte_carlo_result = evaluator.calculate_pos_monte_carlo(
            num_players, samples=10000
        )
        monte_carlo_time = time.time() - start_time
        print(
            f"  Monte Carlo estimation result: {monte_carlo_result}% (Time: {monte_carlo_time:.2f}s)"
        )

        difference = abs(exact_result - monte_carlo_result)
        total_difference += difference
        total_time_difference += exact_time - monte_carlo_time
        print(f"  Difference: {difference:.2f}%\n")

    average_difference = total_difference / 20
    average_time_difference = total_time_difference / 20
    print(f"Average Difference Across All Test Cases: {average_difference:.2f}%")
    print(
        f"Average Time Difference Across All Test Cases: {average_time_difference:.2f}%"
    )
