from enum import Enum
from typing import List
from card import Card, Deck
from player import Player, PlayerAction, PlayerStatus
from hand_evaluator import HandEvaluator


class GamePhase(Enum):
    SETUP = "setup"
    PRE_FLOP = "pre-flop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class PokerGame:
    def __init__(self, players: List[Player], big_blind: int, game_number: int = 0):
        self.players = players
        self.big_blind = big_blind
        self.deck = None
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0  # how much has been raised (includes big blind)
        self.phase = GamePhase.SETUP
        self.button_position = 0
        self.active_player_index = 0
        self.has_played = [False] * len(self.players)
        self.action_history = []
        self.game_number = game_number

    def start_new_hand(self):
        print("\n====== NEW HAND ======")
        # Reset game state
        self.game_number += 1
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.phase = GamePhase.SETUP

        # Reset player_hand statuses
        for i, player in enumerate(self.players):
            player.reset_for_new_hand()
            self.has_played[i] = False if player.status == PlayerStatus.ACTIVE else True

        # Move button to next player_hand
        self.button_position = (self.button_position + 1) % len(self.players)

        # Deal cards
        self._deal_hole_cards()

        # Post blinds
        self._post_blinds()

        # Begin pre-flop betting
        self.phase = GamePhase.PRE_FLOP
        self.active_player_index = (self.button_position + 2) % len(self.players)
        self._adjust_active_player_index()

        # Show game state
        self.display_game_state()

    def _deal_hole_cards(self):
        for player in self.players:
            if player.status != PlayerStatus.OUT:
                player.hole_cards = self.deck.deal(2)

    def _post_blinds(self):
        # Big blind only, no small blind
        bb_position = (self.button_position + 1) % len(self.players)
        bb_player = self.players[bb_position]

        if bb_player.stack > 0:
            action, amount = bb_player.take_action(PlayerAction.BET, self.big_blind)
            self.pot += amount
            self.current_bet = self.big_blind
            print(f"{bb_player.name} posts big blind: {amount}")

    def _adjust_active_player_index(self):
        # Find next active player_hand
        original_index = self.active_player_index
        while not self.players[self.active_player_index].can_make_action():
            self.active_player_index = (self.active_player_index + 1) % len(self.players)
            if self.active_player_index == original_index:
                # We've gone all the way around, no active players
                return False
        return True

    def player_action(self, action: PlayerAction, amount: int = 0) -> bool:
        player = self.players[self.active_player_index]
        amount = min(amount, player.stack)

        # Validate action
        if action == PlayerAction.CHECK and self.current_bet > player.bet_amount:
            print(f"Cannot check when there's a bet. Current bet: {self.current_bet}")
            return False

        if action == PlayerAction.CALL:
            amount = self.current_bet - player.bet_amount

        if action in [PlayerAction.BET, PlayerAction.RAISE]:
            # For a bet, the minimum is the big blind
            min_amount = self.big_blind

            # For a raise, the minimum is the current bet plus the minimum raise amount
            if self.current_bet > 0:
                min_amount = self.current_bet
                action = PlayerAction.RAISE
            else:
                action = PlayerAction.BET

            if amount <= min_amount:
                print(f"Minimum {action.value} is {min_amount}")
                return False

            # Update current bet
            self.current_bet = amount

        if action == PlayerAction.ALL_IN:
            if amount <= 0:
                return False
            if amount > self.current_bet:
                self.current_bet = player.bet_amount

        actual_action, actual_amount = player.take_action(action, amount)
        self.pot += actual_amount
        self.action_history.append((self.phase.value, player.name, actual_action.value, actual_amount))

        # Execute action
        print(f"{player.name} {actual_action.value}s", end="")
        if action in [PlayerAction.BET, PlayerAction.RAISE, PlayerAction.CALL]:
            print(f" {actual_amount}")
        else:
            print()

        # Move to next player_hand
        self.has_played[self.active_player_index] = True
        self.active_player_index = (self.active_player_index + 1) % len(self.players)

        # Check if betting round is complete
        if self.is_betting_round_complete():
            self.advance_game_phase()
        else:
            self._adjust_active_player_index()

        # Show updated game state
        self.display_game_state()
        return True

    def is_betting_round_complete(self) -> bool:
        for player in self.players:
            if player.status in [PlayerStatus.FOLDED, PlayerStatus.ALL_IN]:
                continue
            if player.bet_amount != self.current_bet:
                return False
        return all(self.has_played)

    def advance_game_phase(self):
        # Reset bet amounts for the next betting round

        if self.num_active_players() + self.num_all_in_players() == 1:  # all players folded except one
            self.direct_showdown()  # go directly to showdown and declare winner
            return

        no_one_active = all([p.status in [PlayerStatus.ALL_IN, PlayerStatus.FOLDED] for p in self.players])
        if no_one_active:  # all players other than folded players are all-in
            self.all_in_showdown()  # more than one person are all-in and all others are folded
            return

        for player in self.players:
            player.bet_amount = 0
        self.current_bet = 0

        # Move to the next phase
        if self.phase == GamePhase.PRE_FLOP:
            self.phase = GamePhase.FLOP
            self.community_cards.extend(self.deck.deal(3))
        elif self.phase == GamePhase.FLOP:
            self.phase = GamePhase.TURN
            self.community_cards.extend(self.deck.deal(1))
        elif self.phase == GamePhase.TURN:
            self.phase = GamePhase.RIVER
            self.community_cards.extend(self.deck.deal(1))
        elif self.phase == GamePhase.RIVER:
            self.phase = GamePhase.SHOWDOWN
            self._showdown()
            return

        # Start betting with player_hand after button
        self.active_player_index = (self.button_position + 1) % len(self.players)
        self._adjust_active_player_index()
        self._reset_has_played()

    def direct_showdown(self):  # direct to showdown
        self.phase = GamePhase.SHOWDOWN
        self._showdown()
        return

    def all_in_showdown(
            self):  # if more than one players are all-in first deal sufficient community cards then go to showdown
        num_remaining = 5 - len(self.community_cards)
        if num_remaining > 0:
            self.community_cards.extend(self.deck.deal(num_remaining))

        self.phase = GamePhase.SHOWDOWN
        self._showdown()
        return

    def _showdown(self):
        # Evaluate hands for all players who haven't folded
        active_players = [p for p in self.players if p.status != PlayerStatus.FOLDED]

        if len(active_players) == 1:
            # Only one player_hand left, they win automatically
            winner = active_players[0]
            winner.stack += self.pot
            print(f"{winner.name} wins {self.pot} chips")
            return

        # Evaluate hands
        results = []
        print("\n=== SHOWDOWN ===")
        for player in active_players:
            result = HandEvaluator.evaluate_hand(player.hole_cards, self.community_cards)
            results.append((player, result))

            print(f"{player.name}: {[str(c) for c in player.hole_cards]} - {result.hand_rank.name}")

        # Find the winner(s)
        results.sort(key=lambda x: (x[1].hand_rank.value, x[1].hand_value), reverse=True)
        best_result = results[0][1]

        winners = []
        for player, result in results:
            if (result.hand_rank == best_result.hand_rank and
                    result.hand_value == best_result.hand_value):
                winners.append(player)

        # Distribute the pot
        split_amount = self.pot // len(winners)
        remainder = self.pot % len(winners)

        for player in winners:
            winnings = split_amount
            if remainder > 0:
                winnings += 1
                remainder -= 1

            player.stack += winnings
            print(f"{player.name} wins {winnings} chips with {best_result.hand_rank.name}")

    def display_game_state(self):
        print(f"\nPhase: {self.phase.value}")
        print(f"Pot: {self.pot}")

        if self.community_cards:
            print(f"Community cards: {[str(c) for c in self.community_cards]}")

        print("\nPlayers:")
        for i, player in enumerate(self.players):
            position = ""
            if i == self.button_position:
                position = "(BTN)"

            cards = "[hidden]"
            if player.status == PlayerStatus.FOLDED:
                cards = "folded"
            elif player.status == PlayerStatus.OUT:
                cards = "out"

            active = "→ " if i == self.active_player_index and player.can_make_action() else "  "

            print(f"{active}{player.name} {position}: ${player.stack} {cards} {player.status.value}")

    def _reset_has_played(self):
        self.has_played = [False if player.status == PlayerStatus.ACTIVE else True for player in self.players]

    def num_active_players(self) -> int:
        return len([p for p in self.players if p.status == PlayerStatus.ACTIVE])

    def num_all_in_players(self) -> int:  # players who are all in
        return len([p for p in self.players if p.status == PlayerStatus.ALL_IN])

    def get_player_input(self) -> bool:
        player = self.players[self.active_player_index]
        game_state = self.get_game_state()
        action, amount = player.action(game_state, self.action_history)
        return self.player_action(action, amount)

    def get_game_state(self) -> list[int]:
        """
        Returns the current game state in the following structure:
        <1. Hole Cards' Index>
        card1
        card2
        <2. Community Cards' Index. 0 means not yet dealt>
        card1
        card2
        card3
        card4
        card5
        <3. Pot>
        pot
        <4. Current Raise Amount>
        current_raise
        <5. Blind>
        blind
        <6. Active Player Index>
        active_player_index
        <7. Number of players>
        num_players
        <8. Each player's stack>
        stack1
        stack2
        stack3
        stack4
        <9. Game number>
        game_number
        """
        player = self.players[self.active_player_index]
        player_cards = [card.get_index() for card in player.hole_cards] + (2 - len(player.hole_cards)) * [0]
        community_cards = [card.get_index() for card in self.community_cards] + (5 - len(self.community_cards)) * [0]
        return [
            *player_cards,
            *community_cards,
            self.pot,
            self.current_bet,
            self.big_blind,
            self.active_player_index,
            len(self.players),
            *(p.stack for p in self.players),
            self.game_number,
        ]


if __name__ == "__main__":
    example_players = [
        Player("Alice", 1000),
        Player("Bob", 1000),
        Player("Charlie", 1000),
        Player("Dave", 1000),
    ]
    game = PokerGame(example_players, big_blind=20)
    game.start_new_hand()
    game.advance_game_phase()
    print(game.get_game_state())
