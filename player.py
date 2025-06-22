from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
from card import Card


class PlayerAction(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all-in"


class PlayerStatus(Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all-in"
    OUT = "out"


@dataclass
class Player:
    name: str
    stack: int
    uuid: int = 0
    status: PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: List[Card] = None
    bet_amount: int = 0

    def __post_init__(self):
        if self.hole_cards is None:
            self.hole_cards = []

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.status = PlayerStatus.ACTIVE if self.stack > 0 else PlayerStatus.OUT
        self.bet_amount = 0

    def can_make_action(self) -> bool:
        return self.status in [PlayerStatus.ACTIVE]

    def take_action(self, action: PlayerAction, amount: int = 0) -> Tuple[PlayerAction, int]:
        if action == PlayerAction.FOLD:
            self.status = PlayerStatus.FOLDED
            return action, 0

        if action == PlayerAction.CALL:
            max_bet = min(amount, self.stack)
            self.stack -= max_bet
            self.bet_amount += max_bet
            if self.stack == 0:
                return PlayerAction.ALL_IN, max_bet
            return PlayerAction.CALL, max_bet

        if action in [PlayerAction.BET, PlayerAction.RAISE]:
            # Calculate maximum possible bet
            max_bet = min(amount, self.stack)
            delta = max_bet - self.bet_amount

            if action == PlayerAction.RAISE:
                max_bet = min(amount - self.bet_amount, self.stack)

            # Update stack and bet amount
            if max_bet == self.stack:  # all-in case, when player is all-in the chips he puts in is the stack itself
                self.stack -= max_bet
                self.bet_amount += max_bet
            else:  # when player is not all-in he has to reach maximum bet
                self.stack -= delta  # (max_bet - self.bet_amount)
                self.bet_amount += delta

            # Check if player_hand is all-in
            if self.stack == 0:
                self.status = PlayerStatus.ALL_IN
                return PlayerAction.ALL_IN, max_bet

            return action, delta  # if the case in not all-in case then return delta not max_bet

        if action == PlayerAction.ALL_IN:
            actual = self.stack
            self.bet_amount += self.stack
            self.stack = 0
            self.status = PlayerStatus.ALL_IN
            return PlayerAction.ALL_IN, actual

        return action, 0

    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
        """
        Parameters:
            game_state: a numerical representation of the game state
            action_history: a list of actions taken in sequence

        Returns:
            A PlayerAction and amount tuple
        """
        return PlayerAction.FOLD, 0
