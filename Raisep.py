

import random
import math
from typing import List, Tuple, Optional

from card import Card, Rank, Suit, Deck
from player import Player, PlayerAction
from hand_evaluator import HandEvaluator

_FULL_DECK_FOR_INDEXING = Deck().cards
def card_from_index(index: int) -> Optional[Card]:
    #blahblah
    if 0 <= index < 52:
        for card in _FULL_DECK_FOR_INDEXING:
             if card.get_index() == index:
                 return card
    return None


class AdaptiveRaisePlayer(Player):
    


    RAISE_SIZE_MULTIPLIER = 1.1  
    AGGRESSION_ADJUSTMENT_RATE = 0.07  
    MIN_RAISE_AGGRESSION = 0.4 
    STARTING_AGGRESSION = 0.8 
    BLUFF_TO_VALUE_RATIO = 0.5 

    HISTORY_LENGTH = 15  
    AGGRESSION_THRESHOLD = 0.5 
    RAISEPLAYER_DETECTED_AGGRESSION_BOOST = 0.3

    
    DEFENSIVE_STACK_THRESHOLD_MULT = 1.5  

    GOOD_HAND_WIN_RATE = 0.7 
    PREFLOP_ALLIN_THRESHOLD = 0.95 
    ALLIN_DEVIATION_THRESHOLD = 0.95
    MAX_RAISE_THRESHOLD_MULT = 0.5 

    def __init__(self, name: str, stack: int):
        super().__init__(name, stack)
        self.aggression = self.STARTING_AGGRESSION
        self.opponent_models = {}
        self.my_index = None
        self.wins = 0
        self.total_games = 0
        self.winning_percentage = 0.0
        self.initial_stack = stack

    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
    
        current_raise = game_state[8]
        big_blind = game_state[9]
        self.my_index = game_state[10]
        player_stacks = game_state[12:]
        pot = game_state[7]

  
        self._update_opponent_models(action_history, num_players = game_state[11]) 

        win_rate = self._estimate_hand_strength(game_state, action_history)

        if win_rate > self.ALLIN_DEVIATION_THRESHOLD and random.random() > .95:
            return PlayerAction.ALL_IN, self.stack

        if self.stack > current_raise:
             amount = self._calculate_raise_amount(current_raise, big_blind, pot)
             return PlayerAction.RAISE, amount 
        else:
             return PlayerAction.ALL_IN, self.stack

    def _estimate_hand_strength(self, game_state: list[int], action_history: list) -> float:

        hole_card_indices = game_state[0:2]
        community_card_indices = game_state[2:7]
        hole_cards = [card_from_index(idx) for idx in hole_card_indices if card_from_index(idx)]
        community_cards = [card_from_index(idx) for idx in community_card_indices if card_from_index(idx)]

        if not hole_cards or len(hole_cards) != 2: return 0.2 

  
        hand_evaluator = HandEvaluator()
        hand_result = hand_evaluator.evaluate_hand(hole_cards, community_cards)

        #heheheh
        win_rate = 0
        if hand_result.hand_rank.name == 'ROYAL_FLUSH': win_rate = 0.99
        elif hand_result.hand_rank.name == 'STRAIGHT_FLUSH': win_rate = 0.95
        elif hand_result.hand_rank.name == 'FOUR_OF_A_KIND': win_rate = 0.90
        elif hand_result.hand_rank.name == 'FULL_HOUSE': win_rate = 0.80
        elif hand_result.hand_rank.name == 'FLUSH': win_rate = 0.70
        elif hand_result.hand_rank.name == 'STRAIGHT': win_rate = 0.60
        elif hand_result.hand_rank.name == 'THREE_OF_A_KIND': win_rate = 0.50
        elif hand_result.hand_rank.name == 'TWO_PAIR': win_rate = 0.40
        elif hand_result.hand_rank.name == 'PAIR': win_rate = 0.30
        else: win_rate = 0.20

        return win_rate

    def _calculate_raise_amount(self, current_bet: int, big_blind: int, pot: int) -> int:
      
        raise_amount = int(pot * (1+ random.random())) 
        return min(raise_amount, self.stack)

    def _update_opponent_models(self, action_history: list, num_players: int) -> None:

      for i in range(num_players):
          if i == self.my_index:
              continue
          if i not in self.opponent_models:
              self.opponent_models[i] = {"fold_frequency": 0, "aggressive": False, "actions": []}

          opponent_actions = []
          for item in action_history:
              if isinstance(item, tuple) and len(item) == 3:
                  p, a, _ = item
                  if p == i:
                      opponent_actions.append(a)
              else:
                   print(f"Unexpected action history format: {item}")

          self.opponent_models[i]["actions"].extend(opponent_actions)
          self.opponent_models[i]["actions"] = self.opponent_models[i]["actions"][-self.HISTORY_LENGTH:]

          # Basic fold frequency update
          fold_count = self.opponent_models[i]["actions"].count(PlayerAction.FOLD)
          total_actions = len(self.opponent_models[i]["actions"])
          self.opponent_models[i]["fold_frequency"] = fold_count / total_actions if total_actions > 0 else 0

          # Estimate Aggression: More sophisticated (example)
          aggressive_actions = [PlayerAction.BET, PlayerAction.RAISE, PlayerAction.ALL_IN]
          aggressive_count = sum(1 for a in self.opponent_models[i]["actions"] if a in aggressive_actions)
          aggression_ratio = aggressive_count / total_actions if total_actions > 0 else 0
          self.opponent_models[i]["aggressive"] = aggression_ratio > 0.4  # Example threshold

    def _adjust_aggression(self, opponent_models: dict) -> None:
         """Adjusts the AI's aggression based on opponent tendencies."""
         # Simple Adjustment: If opponents are passive, be more aggressive
         passive_opponents = sum(1 for model in opponent_models.values() if not model["aggressive"])
         if len(opponent_models) > 0 :
            if passive_opponents / len(opponent_models) > 0.6: #If > 60% passive raise aggression
                self.aggression += self.AGGRESSION_ADJUSTMENT_RATE
            else:
                self.aggression -= self.AGGRESSION_ADJUSTMENT_RATE / 2  # Slightly reduce aggression otherwise
         self.aggression = max(self.MIN_RAISE_AGGRESSION, min(self.aggression, 0.9))

    def update_winning_percentage(self, won: bool):
        """Updates the AI's winning percentage."""
        self.total_games += 1
        self.wins += 1 if won else 0
        self.winning_percentage = (self.wins / self.total_games) * 100 if self.total_games > 0 else 0

    def reset_game_stats(self):
        """Resets the game statistics for a new tournament."""
        self.total_games = 0
        self.wins = 0
        self.winning_percentage = 0.0