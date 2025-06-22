# # from player import Player, PlayerAction
# # from hand_evaluator import HandEvaluator
# # import random
# # from card import Card

# # class TournamentAIPlayer(Player):
# #     def __init__(self, name, stack):
# #         super().__init__(name, stack)
# #         self.opponent_models = {}
# #         self.tournament_stage = "early"
# #         self.position = None
# #         self.hand_strength = 0
# #         self.button_position = None  # Initialize button position

# #     def action(self, game_state, action_history):
# #         self.update_game_info(game_state, action_history)
        
# #         # Tournament strategy based on stage
# #         if self.tournament_stage == "early":
# #             return self.early_stage_strategy()
# #         elif self.tournament_stage == "middle":
# #             return self.middle_stage_strategy()
# #         else:
# #             return self.late_stage_strategy()

# #     def update_game_info(self, game_state, action_history):
# #         self.hand_strength = self.evaluate_hand_strength(game_state)
# #         self.position = self.determine_position(game_state)
# #         self.update_opponent_models(action_history)
# #         self.update_tournament_stage(game_state)
# #         self.button_position = game_state[9]  # Assuming button position is available at index 9 in game_state

# #     def evaluate_hand_strength(self, game_state):
# #         hole_cards = [self._index_to_card(game_state[0]), self._index_to_card(game_state[1])]
# #         community_cards = [self._index_to_card(i) for i in game_state[2:7] if i != 0]
# #         return HandEvaluator.evaluate_hand(hole_cards, community_cards).hand_rank.value / 10

# #     def determine_position(self, game_state):
# #         active_player_index = game_state[10]
# #         num_players = game_state[11]
# #         return (active_player_index - self.button_position) % num_players

# #     def update_opponent_models(self, action_history):
# #         for action in action_history:
# #             player_name = action[1]
# #             if player_name not in self.opponent_models:
# #                 self.opponent_models[player_name] = {"aggression": 0.5, "vpip": 0.5}
            
# #             if action[2] in ["raise", "bet"]:
# #                 self.opponent_models[player_name]["aggression"] += 0.1
# #             elif action[2] == "fold":
# #                 self.opponent_models[player_name]["vpip"] -= 0.1

# #     def update_tournament_stage(self, game_state):
# #         avg_stack = sum(game_state[12:12+game_state[11]]) / game_state[11]
# #         if avg_stack > 50 * game_state[9]:  # 50 times the big blind
# #             self.tournament_stage = "early"
# #         elif avg_stack > 25 * game_state[9]:
# #             self.tournament_stage = "middle"
# #         else:
# #             self.tournament_stage = "late"

# #     def early_stage_strategy(self):
# #         if self.hand_strength > 0.8:
# #             return PlayerAction.RAISE, self.calculate_bet_size()
# #         elif self.hand_strength > 0.6 and self.position > 4:  # Late position in early stage
# #             return PlayerAction.CALL, self.calculate_bet_size()
# #         elif self.hand_strength > 0.3:
# #             return PlayerAction.CALL, self.calculate_bet_size()
# #         else:
# #             return PlayerAction.FOLD, 0

# #     def middle_stage_strategy(self):
# #         if self.hand_strength > 0.7:
# #             return PlayerAction.RAISE, self.calculate_bet_size()
# #         elif self.hand_strength > 0.5 and self.position > 3:  # Late position in middle stage
# #             return PlayerAction.CALL, self.calculate_bet_size()
# #         elif self.should_bluff():
# #             return PlayerAction.RAISE, self.calculate_bet_size()
# #         else:
# #             return PlayerAction.FOLD, 0

# #     def late_stage_strategy(self):
# #         if self.hand_strength > 0.6:
# #             return PlayerAction.RAISE, self.calculate_bet_size()
# #         elif self.position > 2:  # Late position in late stage
# #             return PlayerAction.CALL, self.calculate_bet_size()
# #         elif self.should_bluff():
# #             return PlayerAction.ALL_IN, self.stack
# #         else:
# #             return PlayerAction.FOLD, 0

# #     def calculate_bet_size(self):
# #         pot_size = self.get_pot_size()
# #         bet_size = int(pot_size * (0.5 + self.hand_strength))
# #         return min(bet_size, self.stack)

# #     def should_bluff(self):
# #         bluff_threshold = 0.3 - (0.05 * self.position)  # More likely to bluff in later positions
# #         return random.random() < bluff_threshold

# #     def get_pot_size(self):
# #         # Assuming pot size is at index 7 in the game state
# #         return game_state[7]

# #     def _index_to_card(self, index):
# #         suits = ['♠', '♥', '♦', '♣']
# #         ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
# #         rank = ranks[index % 13]  # Get the rank based on index
# #         suit = suits[index // 13]  # Get the suit based on index range
# #         return Card(rank, suit)
# # --- Prepend this to player1.py or ensure these imports are present ---
# import random
# from collections import Counter
# from typing import List, Tuple, Optional

# # Assuming card.py, hand_evaluator.py, game.py are in the same directory or accessible
# from card import Card, Rank, Suit, Deck # Need Deck just for card_from_index helper
# from hand_evaluator import HandEvaluator, HandRank, HandResult
# from player import Player, PlayerAction, PlayerStatus
# from game import GamePhase # Make sure GamePhase is imported if not already

# # --- Helper function to get Card from index (needs Deck structure knowledge) ---
# # You might place this inside the class or keep it global if preferred.
# _FULL_DECK_FOR_INDEXING = Deck().cards # Create a static deck for index lookups
# def card_from_index(index: int) -> Optional[Card]:
#     """Maps an index (0-51) back to a Card object."""
#     if 0 <= index < 52:
#         # This relies on the specific index calculation in Card.get_index()
#         # suit_value = index // 13
#         # rank_value = (index % 13) + 2 # Rank values start from 2
#         # suit = Suit(suit_value)
#         # rank = Rank(rank_value)
#         # return Card(rank, suit)

#         # Simpler approach if we have a static ordered deck
#         for card in _FULL_DECK_FOR_INDEXING:
#              if card.get_index() == index:
#                  return card
#     return None
# # --- End Helper ---

# class TournamentAIPlayer(Player):
#     """
#     An AI player implementing a phase-based strategy with hand evaluation,
#     pot odds considerations, and basic position awareness.
#     """

#     def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
#         """
#         Decides the AI's action based on the game state and history.
#         """
#         # --- 1. Parse Game State ---
#         hole_card_indices = game_state[0:2]
#         community_card_indices = game_state[2:7]
#         pot = game_state[7]
#         current_bet = game_state[8] # The total amount players need to call
#         big_blind = game_state[9]
#         my_index = game_state[10]
#         num_players = game_state[11]
#         player_stacks = game_state[12:12 + num_players]
#         # game_number = game_state[12 + num_players] # Not used in this strategy

#         # --- Get actual Card objects ---
#         self.hole_cards = [card for i in hole_card_indices if (card := card_from_index(i))] # Requires Python 3.8+
#         community_cards = [card for i in community_card_indices if (card := card_from_index(i))]

#         # --- Basic Info ---
#         call_amount = max(0, current_bet - self.bet_amount)
#         effective_stack = self.stack # Simplification: use own stack
#         can_check = (call_amount == 0)

#         # --- Determine Game Phase ---
#         phase = self._get_phase(community_cards)

#         # --- Make Decision based on Phase ---
#         if phase == GamePhase.PRE_FLOP:
#             return self._decide_preflop(call_amount, current_bet, big_blind, num_players, my_index, action_history, player_stacks)
#         else:
#             return self._decide_postflop(phase, community_cards, pot, call_amount, current_bet, big_blind, num_players, my_index, action_history, player_stacks)

#     # --- Helper Methods ---

#     def _get_phase(self, community_cards: List[Card]) -> GamePhase:
#         """Determines the current game phase."""
#         num_community = len(community_cards)
#         if num_community == 0:
#             return GamePhase.PRE_FLOP
#         elif num_community == 3:
#             return GamePhase.FLOP
#         elif num_community == 4:
#             return GamePhase.TURN
#         elif num_community == 5:
#             return GamePhase.RIVER
#         else:
#             # Should not happen in standard Hold'em
#             return GamePhase.SETUP # Or raise an error

#     def _get_effective_position(self, my_index: int, num_players: int, action_history: list, phase: GamePhase) -> str:
#         """Estimates position: 'early', 'middle', 'late', 'blinds'."""
#         # This is complex without knowing the button position.
#         # Simple approximation based on players *yet* to act in *this* round.
        
#         # Find the starting player for this round
#         first_actor_index = -1
#         if phase == GamePhase.PRE_FLOP:
#             # UTG is usually BB + 1 (index + 2 if BB is index + 1)
#             # A very rough estimate based on history if blinds already acted
#              bb_poster_name = None
#              for p_phase, name, act, amt in reversed(action_history):
#                  if p_phase == GamePhase.SETUP.value and act == PlayerAction.BET.value: # Found BB post
#                      bb_poster_name = name
#                      break
#              if bb_poster_name:
#                 try:
#                      # Find index of BB poster (approximation, player list order matters)
#                      bb_index = [p.name for p in self.game.players].index(bb_poster_name) # Needs access to game.players - unavailable here
#                      # So, this position calculation is difficult without more context.
#                      # Let's use a placeholder based purely on index relative to num_players
#                      pass 
#                 except (AttributeError, ValueError):
#                      pass # Fallback if game object or player not found


#         # Placeholder: Very simple relative position
#         if my_index < num_players / 3: return "early"
#         if my_index < 2 * num_players / 3: return "middle"
#         return "late" # Crude, doesn't account for blinds/button well


#     def _classify_preflop_hand(self) -> str:
#         """Classifies starting hand strength."""
#         if not self.hole_cards or len(self.hole_cards) != 2:
#             return "weak" # Should not happen

#         c1, c2 = sorted(self.hole_cards, key=lambda c: c.rank.value, reverse=True)
#         r1, r2 = c1.rank, c2.rank
#         s1, s2 = c1.suit, c2.suit
#         suited = (s1 == s2)
#         vals = sorted([r1.value, r2.value], reverse=True)

#         # Premium
#         if vals[0] == vals[1] and vals[0] >= Rank.JACK.value: return "premium" # JJ+
#         if vals == [Rank.ACE.value, Rank.KING.value]: return "premium" # AK

#         # Strong
#         if vals[0] == vals[1] and vals[0] >= Rank.NINE.value : return "strong" # 99, TT
#         if suited and vals[0] == Rank.ACE.value and vals[1] >= Rank.JACK.value: return "strong" # AJs, AQs
#         if suited and vals == [Rank.KING.value, Rank.QUEEN.value]: return "strong" # KQs

#         # Good
#         if vals[0] == vals[1]: return "good" # 22-88
#         if suited and vals[0] == Rank.ACE.value: return "good" # A2s-ATs
#         if suited and vals[0] == Rank.KING.value and vals[1] >= Rank.TEN.value: return "good" # KTs, KJs
#         if suited and vals[0] == Rank.QUEEN.value and vals[1] >= Rank.TEN.value: return "good" # QTs, QJs
#         if suited and vals[0] == Rank.JACK.value and vals[1] >= Rank.NINE.value: return "good" # J9s, JTs
#         if suited and vals[0] == Rank.TEN.value and vals[1] == Rank.NINE.value: return "good" # T9s
#         if not suited and vals == [Rank.ACE.value, Rank.QUEEN.value]: return "good" # AQo
#         if not suited and vals == [Rank.ACE.value, Rank.JACK.value]: return "good" # AJo

#         # Speculative
#         if suited and vals[1] >= Rank.FIVE.value and (vals[0] - vals[1] <= 2): return "speculative" # Suited connectors/gappers like 98s, 76s, T8s
#         if not suited and vals == [Rank.KING.value, Rank.QUEEN.value]: return "speculative" # KQo
#         if not suited and vals == [Rank.ACE.value, Rank.TEN.value]: return "speculative" # ATo

#         return "weak"

#     def _calculate_raise_size(self, current_bet: int, big_blind: int, pot: int, num_limpers: int = 0) -> int:
#         """Calculates a reasonable raise amount."""
#         if current_bet == 0: # Opening raise
#             size = big_blind * 3 + (num_limpers * big_blind)
#         else: # Re-raise (3bet+)
#             # Simplified: Pot size raise + previous bet amount
#             size = pot + current_bet # Roughly pot-sized raise accounting for call
#             # Common 3-bet sizing: ~3x the previous raise size
#             # size = current_bet * 3 # Another simple approach
            
#         # Ensure minimum raise rules are met (at least double the last bet/raise delta)
#         # This needs more history context, simplifying here
#         min_raise = current_bet + big_blind # Simplistic minimum if opening
#         if current_bet > 0:
#            # Find last raise amount... requires parsing history, complex here
#            # Assume minimum raise is at least the size of the last bet/raise
#            min_raise = current_bet * 2 # Rough estimate, not strictly correct rule

#         size = max(size, min_raise)

#         # Don't bet more than stack
#         return min(size, self.stack + self.bet_amount) # Target total bet amount


#     def _decide_preflop(self, call_amount: int, current_bet: int, big_blind: int, num_players: int, my_index: int, action_history: list, player_stacks: list) -> Tuple[PlayerAction, int]:
#         """Makes the pre-flop decision."""
#         strength = self._classify_preflop_hand()
#         can_check = (call_amount == 0)
#         # position = self._get_effective_position(my_index, num_players, action_history, GamePhase.PRE_FLOP) # Position estimation is hard here

#         # --- Short Stack Strategy ---
#         if self.stack <= big_blind * 15:
#             if strength in ["premium", "strong"] or (strength == "good" and self.hole_cards[0].rank == self.hole_cards[1].rank): # Push pairs+ and strong aces
#                  return PlayerAction.ALL_IN, self.stack
#             else:
#                  return PlayerAction.FOLD, 0 if not can_check else PlayerAction.CHECK, 0

#         # --- Regular Stack Strategy ---
#         num_limpers = 0 # TODO: Calculate from action_history if needed for sizing
#         pot_approx = sum(p[3] for p in action_history if p[0] == GamePhase.PRE_FLOP.value) + big_blind # Rough pot

#         if can_check:
#             if strength in ["premium", "strong"]:
#                 amount = self._calculate_raise_size(0, big_blind, pot_approx, num_limpers)
#                 return PlayerAction.BET, amount # Use BET when opening
#             elif strength in ["good", "speculative"]:
#                  # Check is fine, maybe raise late position sometimes
#                  return PlayerAction.CHECK, 0
#             else: # Weak
#                  return PlayerAction.CHECK, 0
#         else: # Must call, raise, or fold
#             if strength == "premium":
#                 # Re-raise (3-bet or more)
#                 amount = self._calculate_raise_size(current_bet, big_blind, pot_approx)
#                 # If raise is effectively all-in, just go all-in
#                 if amount >= self.stack + self.bet_amount:
#                     return PlayerAction.ALL_IN, self.stack
#                 return PlayerAction.RAISE, amount
#             elif strength == "strong":
#                 # Consider calling or 3-betting depending on raise size and position
#                 # Simple: Call moderate raises, fold to huge ones, maybe 3bet sometimes
#                 if call_amount < self.stack * 0.2: # Call if less than 20% of stack
#                     return PlayerAction.CALL, call_amount
#                 else:
#                     return PlayerAction.FOLD, 0 # Fold to large bets
#             elif strength == "good":
#                  # Call smaller raises, especially pairs and suited aces
#                  if call_amount <= big_blind * 3 and call_amount < self.stack * 0.1: # Call small raises
#                      return PlayerAction.CALL, call_amount
#                  else:
#                      return PlayerAction.FOLD, 0
#             elif strength == "speculative":
#                   # Only call if very cheap and multiway pot implied odds
#                   # Simple: Fold to any raise
#                   return PlayerAction.FOLD, 0
#             else: # Weak
#                 return PlayerAction.FOLD, 0

#     def _evaluate_postflop_hand(self, community_cards: List[Card]) -> Tuple[HandResult, Optional[str], int]:
#         """Evaluates hand strength, identifies draws, and counts outs."""
#         if not self.hole_cards: return HandEvaluator.evaluate_hand([], community_cards), None, 0

#         best_eval = HandEvaluator.evaluate_hand(self.hole_cards, community_cards)
#         all_cards = self.hole_cards + community_cards

#         # Draw Detection
#         draw_type = None
#         outs = 0
#         ranks = sorted([c.rank.value for c in all_cards])
#         suits = [c.suit for c in all_cards]
#         suit_counts = Counter(suits)
#         rank_counts = Counter(ranks)

#         # Flush Draw
#         flush_suit = None
#         for suit, count in suit_counts.items():
#             if count == 4:
#                 draw_type = "flush_draw"
#                 flush_suit = suit
#                 # Count outs (cards of the flush suit not already visible)
#                 known_suits = len([c for c in all_cards if c.suit == flush_suit])
#                 total_in_suit = 13
#                 outs = total_in_suit - known_suits
#                 break # Assume only one flush draw possible for simplicity

#         # Straight Draw - More complex
#         unique_ranks = sorted(list(set(ranks)))
#         # Add Ace low possibility
#         if Rank.ACE.value in unique_ranks:
#              unique_ranks_ace_low = sorted([1 if r == Rank.ACE.value else r for r in unique_ranks])
#              if unique_ranks != unique_ranks_ace_low : # Avoid duplicates if only Ace high
#                   unique_ranks_ace_low = sorted(list(set(unique_ranks_ace_low)))
#                   potential_straights_low = self._find_potential_straights(unique_ranks_ace_low)

#         potential_straights = self._find_potential_straights(unique_ranks)


#         # Check OESD (Open-Ended Straight Draw) - 4 consecutive ranks
#         is_oesd = False
#         for i in range(len(unique_ranks) - 3):
#              if unique_ranks[i+3] - unique_ranks[i] == 3: # e.g., 5,6,7,8 needs 4 or 9
#                  is_oesd = True
#                  current_outs = 8 # Typically 8 outs
#                  # Handle edge cases (e.g., A234, TJQK needs specific outs)
#                  low_card = unique_ranks[i]
#                  high_card = unique_ranks[i+3]
#                  if low_card == 2 and high_card == 5 : # A2345 case needs Ace or 6
#                       # This requires checking if Ace is one of the 4, complex
#                       pass # Simplified: Assume 8 outs generally
#                  elif low_card == Rank.TEN.value and high_card == Rank.KING.value: #TJQK needs 9 or A
#                      pass # Simplified: Assume 8 outs

#                  if draw_type == "flush_draw": draw_type = "combo_draw"
#                  else: draw_type = "straight_draw"
#                  outs += current_outs
#                  break # Prioritize OESD if found

#         # Check Gutshot Straight Draw - 4 ranks with one gap
#         if not is_oesd:
#              for i in range(len(unique_ranks) - 3):
#                   # Example: 5, 6, 8, 9 needs a 7 (gap = 1)
#                   four_ranks = unique_ranks[i:i+4]
#                   diffs = [four_ranks[j+1] - four_ranks[j] for j in range(3)]
#                   if diffs.count(1) == 2 and diffs.count(2) == 1: # One gap of size 2
#                        is_gutshot = True
#                        current_outs = 4
#                        if draw_type == "flush_draw": draw_type = "combo_draw"
#                        else: draw_type = "straight_draw"
#                        outs += current_outs
#                        break # Found a gutshot

#         # Recalculate outs for Combo Draw - subtract overlapping cards
#         if draw_type == "combo_draw":
#             # Find cards that complete both straight and flush
#             straight_completers = [] # Need to know which ranks complete straight
#             flush_completers = [Card(Rank(r), flush_suit) for r in range(2, 15)] # All ranks of flush suit
#             # This overlap calculation is tricky, simplify: just add outs? No, overcounts.
#             # Approximation: Max(flush_outs, straight_outs) + Min(...) - overlap? Too complex for now.
#             # Simple combo: outs = flush_outs + straight_outs (overcounts, but aggressive)
#             # Let's stick with the sum for simplicity, knowing it's high.
#         pass 


#         return best_eval, draw_type, outs

#     def _find_potential_straights(self, unique_ranks):
#         # Helper for straight draw detection - finds sequences
#         sequences = []
#         if not unique_ranks: return sequences
#         current_sequence = [unique_ranks[0]]
#         for i in range(1, len(unique_ranks)):
#             if unique_ranks[i] == unique_ranks[i-1] + 1:
#                 current_sequence.append(unique_ranks[i])
#             else:
#                 if len(current_sequence) >= 3: # Need at least 3 for potential
#                     sequences.append(current_sequence)
#                 current_sequence = [unique_ranks[i]]
#         if len(current_sequence) >= 3:
#             sequences.append(current_sequence)
#         return sequences


#     def _calculate_pot_odds_percentage(self, pot: int, call_amount: int) -> float:
#         """Calculates pot odds as a percentage required equity."""
#         if pot + call_amount == 0: return 100.0 # Avoid division by zero
#         return (call_amount / (pot + call_amount + call_amount)) * 100 # Add own call to pot

#     def _calculate_draw_equity_percentage(self, outs: int, phase: GamePhase) -> float:
#         """Estimates draw equity using Rule of 2 and 4."""
#         if phase == GamePhase.FLOP:
#             return outs * 4 # Approx equity for turn + river
#         elif phase == GamePhase.TURN:
#             return outs * 2 # Approx equity for river
#         else: # River or Preflop (not applicable)
#             return 0.0

#     def _decide_postflop(self, phase: GamePhase, community_cards: List[Card], pot: int, call_amount: int, current_bet: int, big_blind: int, num_players: int, my_index: int, action_history: list, player_stacks: list) -> Tuple[PlayerAction, int]:
#         """Makes the post-flop decision."""

#         hand_eval, draw_type, outs = self._evaluate_postflop_hand(community_cards)
#         rank = hand_eval.hand_rank
#         can_check = (call_amount == 0)
        
#         # --- Strength Assessment ---
#         is_strong_made = rank.value >= HandRank.TWO_PAIR.value # Two pair or better
#         is_medium_made = rank == HandRank.PAIR # Only pairs
#         is_draw = draw_type is not None
#         is_garbage = not is_strong_made and not is_medium_made and not is_draw

#         # --- Pot Odds & Equity ---
#         pot_odds_pct = self._calculate_pot_odds_percentage(pot, call_amount)
#         draw_equity_pct = self._calculate_draw_equity_percentage(outs, phase)

#         # --- Decision Logic ---

#         if can_check:
#             if is_strong_made:
#                 # Value bet
#                 amount = self._calculate_bet_size(pot, phase, is_value=True)
#                 return PlayerAction.BET, amount
#             elif is_medium_made:
#                 # Check, maybe bet small on dry boards vs few opponents
#                 return PlayerAction.CHECK, 0
#             elif is_draw:
#                 # Check, maybe semi-bluff
#                 # Simple: Check draws when checked to
#                 return PlayerAction.CHECK, 0
#             else: # Garbage
#                 # Check, maybe bluff C-Bet on flop if preflop aggressor
#                 # Need to know if AI was PFA - check action_history
#                 was_preflop_aggressor = False # TODO: Implement check
#                 if phase == GamePhase.FLOP and was_preflop_aggressor:
#                      # Simple C-Bet bluff
#                      amount = self._calculate_bet_size(pot, phase, is_value=False)
#                      return PlayerAction.BET, amount
#                 else:
#                      return PlayerAction.CHECK, 0
#         else: # Facing a bet
#             if is_strong_made:
#                  # Value raise? Call?
#                  # Simple: Call if bet is reasonable, raise if very strong (set+)
#                  if rank.value >= HandRank.THREE_OF_A_KIND.value:
#                        amount = self._calculate_raise_size(current_bet, big_blind, pot)
#                        if amount >= self.stack + self.bet_amount: return PlayerAction.ALL_IN, self.stack
#                        return PlayerAction.RAISE, amount
#                  else: # Two pair, maybe top pair
#                       if call_amount < self.stack * 0.3: # Call reasonable bets
#                           return PlayerAction.CALL, call_amount
#                       else:
#                           return PlayerAction.FOLD, 0 # Fold to large bets
#             elif is_medium_made:
#                 # Bluff catch? Depends on odds and opponent.
#                 # Simple: Call small bets, fold to large ones. Check pot odds vs perceived bluff freq (hard)
#                 if pot_odds_pct < 25 and call_amount < self.stack * 0.1: # Call very small bets cheaply
#                     return PlayerAction.CALL, call_amount
#                 else:
#                     return PlayerAction.FOLD, 0
#             elif is_draw:
#                 # Call if odds are good, maybe raise as semi-bluff
#                 if draw_equity_pct >= pot_odds_pct:
#                     # Mathematically correct to call
#                     return PlayerAction.CALL, call_amount
#                 else:
#                      # Consider semi-bluff raise if draw is strong (OESD/Flush/Combo)
#                      if outs >= 8 and phase != GamePhase.RIVER:
#                          # Semi-bluff raise
#                           amount = self._calculate_raise_size(current_bet, big_blind, pot)
#                           if amount >= self.stack + self.bet_amount: return PlayerAction.ALL_IN, self.stack
#                           # Make semi-bluff smaller than value raise?
#                           amount = min(amount, pot + current_bet) # Cap raise size slightly
#                           return PlayerAction.RAISE, amount
#                      else:
#                           # Odds not good enough, draw not strong enough to bluff
#                           return PlayerAction.FOLD, 0
#             else: # Garbage
#                 return PlayerAction.FOLD, 0

#     def _calculate_bet_size(self, pot: int, phase: GamePhase, is_value: bool) -> int:
#         """ Calculates a reasonable post-flop bet size. """
#         percentage = 0.0
#         if is_value:
#             # Bet larger with stronger hands / on later streets
#             if phase == GamePhase.FLOP: percentage = 0.50 # 1/2 pot
#             elif phase == GamePhase.TURN: percentage = 0.66 # 2/3 pot
#             elif phase == GamePhase.RIVER: percentage = 0.75 # 3/4 pot
#         else: # Bluff / Semi-bluff
#             if phase == GamePhase.FLOP: percentage = 0.40 # Smaller C-bet
#             else: percentage = 0.60 # Larger bluff on later streets? Risky.

#         amount = int(pot * percentage)

#         # Ensure minimum bet (big blind) and don't bet more than stack
#         amount = max(amount, self.game.big_blind if hasattr(self, 'game') else 20) # Use BB if game accessible, else default
#         amount = min(amount, self.stack)
#         return amount


# # --- Add the following to your main.py or wherever you instantiate players ---
# # Make sure the TournamentAIPlayer class definition above is used.
# # Example instantiation:
# # players = [
# #     InputPlayer("Alice", 1000),
# #     TournamentAIPlayer("Bot Bob", 1000), # Use the new AI
# #     InputPlayer("Charlie", 1000),
# #     TournamentAIPlayer("Bot David", 1000),
# # ]

# --- player1.py ---

import random
from collections import Counter
from typing import List, Tuple, Optional

# Assuming card.py, hand_evaluator.py, game.py are accessible
from card import Card, Rank, Suit, Deck
from hand_evaluator import HandEvaluator, HandRank, HandResult
from player import Player, PlayerAction, PlayerStatus
from game import GamePhase

# --- Helper function to get Card from index ---
_FULL_DECK_FOR_INDEXING = Deck().cards
def card_from_index(index: int) -> Optional[Card]:
    """Maps an index (0-51) back to a Card object."""
    if 0 <= index < 52:
        for card in _FULL_DECK_FOR_INDEXING:
             if card.get_index() == index:
                 return card
    return None
# --- End Helper ---

class TournamentAIPlayer(Player):
    """
    An AI player implementing a phase-based strategy with hand evaluation,
    pot odds considerations, basic position awareness,
    **AND ADDED RANDOM AGGRESSION.**
    """
    # --- Configuration for Random Aggression ---
    # Adjust these probabilities (0.0 to 1.0) to tune aggression level
    PREFLOP_RANDOM_OPEN_CHANCE_LP = 0.15  # Chance to open raise weak/speculative in LP
    PREFLOP_RANDOM_3BET_CHANCE_GOOD = 0.20 # Chance to 3bet 'Good' hands instead of call/fold
    PREFLOP_RANDOM_3BET_CHANCE_SPEC = 0.10 # Chance to 3bet 'Speculative' as bluff (LP mostly)

    POSTFLOP_RANDOM_BET_CHECKED_MEDIUM = 0.25 # Chance to bet medium pairs when checked to
    POSTFLOP_RANDOM_BET_CHECKED_DRAW = 0.30   # Chance to semi-bluff bet draws when checked to
    POSTFLOP_RANDOM_BET_CHECKED_GARBAGE = 0.10 # Chance to bluff bet garbage when checked to (Flop mostly)

    POSTFLOP_RANDOM_RAISE_FACING_BET_MEDIUM = 0.15 # Chance to bluff-raise medium pairs vs bet
    POSTFLOP_RANDOM_RAISE_FACING_BET_DRAW = 0.25   # Chance to semi-bluff raise draws vs bet (even w/o odds)
    # --- End Configuration ---


    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
        """
        Decides the AI's action based on the game state and history.
        """
        # --- 1. Parse Game State ---
        hole_card_indices = game_state[0:2]
        community_card_indices = game_state[2:7]
        pot = game_state[7]
        current_bet = game_state[8] # The total amount players need to call
        big_blind = game_state[9]
        my_index = game_state[10]
        num_players = game_state[11]
        player_stacks = game_state[12:12 + num_players]

        # --- Get actual Card objects ---
        self.hole_cards = [card for i in hole_card_indices if (card := card_from_index(i))]
        if not self.hole_cards or len(self.hole_cards) < 2 : # Safety check
             print(f"AI {self.name} received invalid hole cards: {hole_card_indices}")
             call_amount_safe = max(0, current_bet - self.bet_amount)
             return PlayerAction.FOLD, 0 if call_amount_safe > 0 else PlayerAction.CHECK, 0

        community_cards = [card for i in community_card_indices if (card := card_from_index(i))]

        # --- Basic Info ---
        call_amount = max(0, current_bet - self.bet_amount)
        can_check = (call_amount == 0)
        phase = self._get_phase(community_cards)

        # --- Store Big Blind for helpers ---
        self._current_bb = big_blind # Store BB temporarily for sizing functions

        # --- Make Decision based on Phase ---
        if phase == GamePhase.PRE_FLOP:
            return self._decide_preflop(call_amount, current_bet, big_blind, num_players, my_index, action_history, player_stacks, pot)
        else:
            return self._decide_postflop(phase, community_cards, pot, call_amount, current_bet, big_blind, num_players, my_index, action_history, player_stacks)


    def _get_phase(self, community_cards: List[Card]) -> GamePhase:
        """Determines the current game phase."""
        # (Keep existing implementation)
        num_community = len(community_cards)
        if num_community == 0: return GamePhase.PRE_FLOP
        elif num_community == 3: return GamePhase.FLOP
        elif num_community == 4: return GamePhase.TURN
        elif num_community == 5: return GamePhase.RIVER
        else: return GamePhase.SETUP

    def _get_effective_position(self, my_index: int, num_players: int, action_history: list, phase: GamePhase) -> str:
         """Approximates position. Crude without button knowledge."""
         # (Keep existing implementation)
         relative_pos = my_index / float(num_players) if num_players > 0 else 0
         if relative_pos < 0.33: return "early"
         if relative_pos < 0.66: return "middle"
         return "late"

    def _classify_preflop_hand(self) -> str:
        """ Classifies starting hand strength. """
        # (Keep existing aggressive implementation)
        if not self.hole_cards or len(self.hole_cards) != 2: return "weak"
        c1, c2 = sorted(self.hole_cards, key=lambda c: c.rank.value, reverse=True)
        r1, r2 = c1.rank, c2.rank
        suited = (c1.suit == c2.suit)
        vals = sorted([r1.value, r2.value], reverse=True)
        is_pair = (vals[0] == vals[1])

        if is_pair and vals[0] >= Rank.JACK.value: return "premium"
        if vals == [Rank.ACE.value, Rank.KING.value]: return "premium"
        if is_pair and vals[0] >= Rank.NINE.value : return "strong"
        if suited and vals[0] == Rank.ACE.value and vals[1] >= Rank.JACK.value: return "strong"
        if suited and vals == [Rank.KING.value, Rank.QUEEN.value]: return "strong"
        if not suited and vals == [Rank.ACE.value, Rank.QUEEN.value]: return "strong"
        if is_pair: return "good"
        if suited:
            if vals[0] == Rank.ACE.value: return "good"
            if vals[0] == Rank.KING.value and vals[1] >= Rank.TEN.value: return "good"
            if vals[0] == Rank.QUEEN.value and vals[1] >= Rank.TEN.value: return "good"
            if vals[0] == Rank.JACK.value and vals[1] >= Rank.NINE.value: return "good"
            if vals[0] == Rank.TEN.value and vals[1] >= Rank.NINE.value: return "good"
            if vals[0] == Rank.NINE.value and vals[1] >= Rank.EIGHT.value: return "good"
            if vals[0] == Rank.EIGHT.value and vals[1] >= Rank.SEVEN.value: return "good"
            if vals[0] == Rank.SEVEN.value and vals[1] >= Rank.SIX.value: return "good"
        if not suited:
             if vals[0] == Rank.ACE.value and vals[1] >= Rank.TEN.value: return "good"
             if vals == [Rank.KING.value, Rank.QUEEN.value]: return "good"
        if suited and vals[1] >= Rank.FIVE.value and (vals[0] - vals[1] <= 2): return "speculative"
        if not suited and vals[0] == Rank.KING.value and vals[1] == Rank.JACK.value: return "speculative"
        return "weak"


    def _calculate_raise_amount(self, current_bet: int, big_blind: int, pot: int, num_players_in_hand: int = 1, is_3bet: bool = False, is_value: bool = True) -> int:
        """
        Calculates the amount TO RAISE BY (not the total bet size).
        Returns 0 if raise is not possible or doesn't make sense.
        """
        call_amount = max(0, current_bet - self.bet_amount)
        min_raise_increase = big_blind # Default minimum increase

        if current_bet == 0: # Opening raise
            # Standard open: 3x BB
            target_total_bet = big_blind * 3
            # TODO: Add for limpers if info available
        else: # Re-raise (3bet+)
            # Calculate minimum legal raise increase (last bet/raise amount)
            # This is hard without full history parsing, use BB as proxy floor for increase
            last_bet_increase = max(big_blind, current_bet - 0) # Simplification
            min_raise_increase = max(min_raise_increase, last_bet_increase)

            # Calculate target total bet size (e.g., ~3x last bet, or Pot Size Raise)
            # Pot size raise: Call Amount + (Pot + Call Amount) = Pot + 2*CallAmount increase
            # Simpler: Target ~3x the current bet level for 3bet/raises
            target_total_bet = current_bet * 3
            if not is_value: # Make bluff/semi-bluff raises slightly smaller?
                 target_total_bet = current_bet * 2.5

        # Ensure target total is at least minimum legal total raise
        min_legal_total_bet = current_bet + min_raise_increase
        target_total_bet = max(target_total_bet, min_legal_total_bet)

        # Calculate the actual amount *to raise by*
        raise_amount = target_total_bet - self.bet_amount

        # Ensure it's affordable and actually a raise
        raise_amount = min(raise_amount, self.stack) # Cannot raise more than stack
        raise_amount = max(0, raise_amount) # Cannot be negative

        # Check if the calculated raise is actually more than just calling
        if raise_amount <= call_amount and self.stack > call_amount:
            # If calculated raise is too small, try minimum legal raise increase if possible
            min_legal_raise_amount = (current_bet + min_raise_increase) - self.bet_amount
            min_legal_raise_amount = min(min_legal_raise_amount, self.stack)
            if min_legal_raise_amount > call_amount:
                 raise_amount = min_legal_raise_amount
            else:
                 # Cannot legally raise more than a call, return 0 to indicate failure
                 return 0

        # Final check: is it still a raise > call amount?
        if raise_amount <= call_amount:
            return 0 # Failed to calculate a valid raise amount

        return int(raise_amount) # Return the amount to add

    def _calculate_bet_amount(self, pot: int, big_blind: int, is_value: bool = True, phase: GamePhase = GamePhase.FLOP) -> int:
        """ Calculates the amount TO BET (when opening action post-flop or pre-flop)."""
        percentage = 0.0
        if is_value:
            if phase == GamePhase.FLOP: percentage = random.uniform(0.5, 0.7)
            elif phase == GamePhase.TURN: percentage = random.uniform(0.6, 0.8)
            elif phase == GamePhase.RIVER: percentage = random.uniform(0.65, 1.0)
            else: percentage = 0.6 # Default for preflop open? No, use raise calc
        else: # Bluff / Semi-bluff
            if phase == GamePhase.FLOP: percentage = random.uniform(0.4, 0.6)
            else: percentage = random.uniform(0.5, 0.7)

        amount = int(pot * percentage)
        amount = max(amount, big_blind) # Min bet is BB
        amount = min(amount, self.stack) # Max bet is stack
        return amount

    def _decide_preflop(self, call_amount: int, current_bet: int, big_blind: int, num_players: int, my_index: int, action_history: list, player_stacks: list, pot: int) -> Tuple[PlayerAction, int]:
        """Makes the pre-flop decision - WITH RANDOM AGGRESSION."""
        strength = self._classify_preflop_hand()
        can_check = (call_amount == 0)
        position = self._get_effective_position(my_index, num_players, action_history, GamePhase.PRE_FLOP)

        # --- Short Stack Strategy (Remains Push/Fold, maybe slightly wider) ---
        if self.stack <= big_blind * 15:
             should_push = strength in ["premium", "strong"] or \
                           (strength == "good" and (self.hole_cards[0].rank == self.hole_cards[1].rank or self.hole_cards[0].rank.value >= Rank.NINE.value)) or \
                           (strength == "speculative" and self.hole_cards[0].suit == self.hole_cards[1].suit and self.hole_cards[0].rank.value >= Rank.SEVEN.value) # Push wider suited stuff
             if should_push: return PlayerAction.ALL_IN, self.stack
             else: return (PlayerAction.CHECK, 0) if can_check else (PlayerAction.FOLD, 0)

        # --- Regular Stack - Random Aggression Strategy ---
        num_players_active = len([s for s in player_stacks if s > 0]) # Crude active count

        # --- A: Action is on AI (Can Check or Bet) ---
        if can_check:
            should_raise = False
            if strength in ["premium", "strong"]: should_raise = True
            elif strength == "good":
                if position != "early" or random.random() < 0.7: should_raise = True
            # RANDOM OPEN: Chance to raise weaker hands from LP
            elif strength in ["speculative", "weak"] and position == "late" and random.random() < self.PREFLOP_RANDOM_OPEN_CHANCE_LP:
                 should_raise = True

            if should_raise:
                # Use _calculate_bet_amount for opening preflop? Or stick to raise logic?
                # Let's use raise logic targeting 3x BB
                target_total_bet = big_blind * 3 # Simple open size target
                bet_amount = min(target_total_bet, self.stack)
                bet_amount = max(bet_amount, big_blind) # Ensure min bet
                return PlayerAction.BET, bet_amount
            else:
                 return PlayerAction.CHECK, 0

        # --- B: Facing a Bet/Raise ---
        else:
            should_raise = False
            should_call = False
            raise_is_value = True # Assume value unless it's a random bluff

            if strength in ["premium", "strong"]:
                should_raise = True
            # RANDOM 3BET: Chance to 3bet 'Good' hands
            elif strength == "good" and random.random() < self.PREFLOP_RANDOM_3BET_CHANCE_GOOD:
                 should_raise = True
                 raise_is_value = False # Semi-bluff 3bet
            # RANDOM 3BET BLUFF: Chance to 3bet 'Speculative'/'Weak' from LP
            elif strength in ["speculative", "weak"] and position == "late" and random.random() < self.PREFLOP_RANDOM_3BET_CHANCE_SPEC:
                 should_raise = True
                 raise_is_value = False # Pure bluff 3bet
            else: # Default logic for 'Good'/'Speculative' if not randomly raising
                 is_pair = self.hole_cards[0].rank == self.hole_cards[1].rank
                 is_suited_ace = self.hole_cards[0].rank == Rank.ACE and self.hole_cards[0].suit == self.hole_cards[1].suit
                 if strength == "good" and (is_pair or is_suited_ace) and call_amount < self.stack * 0.1 and call_amount <= big_blind * 4:
                       should_call = True # Narrow calling range remains

            # Execute Raise Decision
            if should_raise:
                raise_amount = self._calculate_raise_amount(current_bet, big_blind, pot, num_players_active, is_3bet=True, is_value=raise_is_value)

                if raise_amount >= self.stack: # If raise is effectively all-in
                    return PlayerAction.ALL_IN, self.stack
                elif raise_amount > 0: # If valid raise amount calculated
                     return PlayerAction.RAISE, raise_amount
                else: # Raise calculation failed (e.g., couldn't make min raise)
                     # Fallback: Call if we have some 'good' hand equity, else fold
                     if strength == "good" and call_amount <= self.stack:
                          should_call = True
                     else:
                          return PlayerAction.FOLD, 0

            # Execute Call Decision (only if not raising)
            if should_call:
                 if call_amount <= self.stack:
                     return PlayerAction.CALL, call_amount
                 else:
                      return PlayerAction.FOLD, 0 # Cannot afford call

            # Default: Fold if not raising or calling
            return PlayerAction.FOLD, 0

    def _evaluate_postflop_hand(self, community_cards: List[Card]) -> Tuple[HandResult, Optional[str], int]:
        """Evaluates hand strength, identifies draws, and counts outs."""
        # (Keep existing implementation)
        if not self.hole_cards: return HandEvaluator.evaluate_hand([], community_cards), None, 0
        best_eval = HandEvaluator.evaluate_hand(self.hole_cards, community_cards)
        all_cards = self.hole_cards + community_cards
        draw_type = None
        outs = 0
        ranks = sorted([c.rank.value for c in all_cards])
        suits = [c.suit for c in all_cards]
        suit_counts = Counter(suits)
        flush_suit = None
        for suit, count in suit_counts.items():
            if count == 4:
                draw_type = "flush_draw"
                flush_suit = suit
                known_suits = len([c for c in all_cards if c.suit == flush_suit])
                outs = 13 - known_suits
                break
        unique_ranks = sorted(list(set(ranks)))
        is_oesd = False; is_gutshot = False; straight_outs = 0
        for i in range(len(unique_ranks) - 3):
             sequence = unique_ranks[i:i+4]
             if all(sequence[j] == sequence[0] + j for j in range(4)):
                 is_oesd = True; straight_outs = max(straight_outs, 8); break
        if not is_oesd:
            unique_ranks_ace_low = sorted(list(set([1 if r == Rank.ACE.value else r for r in ranks])))
            potential_sets = [unique_ranks, unique_ranks_ace_low] if 1 in unique_ranks_ace_low else [unique_ranks]
            for rank_set in potential_sets:
                 if is_gutshot: break
                 for i in range(len(rank_set) - 4):
                      window = rank_set[i:i+5]
                      for j in range(5):
                          four_cards = sorted(window[:j] + window[j+1:])
                          if len(set(four_cards)) == 4 and (four_cards[3] - four_cards[0] == 4 or four_cards[3] - four_cards[0] == 3): # Check connectivity
                              if four_cards[3] - four_cards[0] == 3: continue # Made straight
                              diffs = [four_cards[k+1]-four_cards[k] for k in range(3)]
                              if diffs.count(1) == 2 and diffs.count(2) == 1:
                                  is_gutshot = True; straight_outs = max(straight_outs, 4); break
                      if is_gutshot: break
        if straight_outs > 0:
            if draw_type == "flush_draw": draw_type = "combo_draw"; outs += straight_outs
            else: draw_type = "straight_draw"; outs = straight_outs
        return best_eval, draw_type, outs


    def _calculate_pot_odds_percentage(self, pot: int, call_amount: int) -> float:
        """Calculates pot odds as a percentage required equity."""
        # (Keep existing implementation)
        if call_amount <= 0: return 0.0
        total_pot_after_call = pot + call_amount + call_amount
        if total_pot_after_call == 0: return 100.0
        return (call_amount / total_pot_after_call) * 100

    def _calculate_draw_equity_percentage(self, outs: int, phase: GamePhase) -> float:
        """Estimates draw equity using Rule of 2 and 4."""
        # (Keep existing implementation)
        if phase == GamePhase.FLOP: return outs * 4
        elif phase == GamePhase.TURN: return outs * 2
        else: return 0.0

    def _decide_postflop(self, phase: GamePhase, community_cards: List[Card], pot: int, call_amount: int, current_bet: int, big_blind: int, num_players: int, my_index: int, action_history: list, player_stacks: list) -> Tuple[PlayerAction, int]:
        """Makes the post-flop decision - WITH RANDOM AGGRESSION."""

        hand_eval, draw_type, outs = self._evaluate_postflop_hand(community_cards)
        rank = hand_eval.hand_rank
        can_check = (call_amount == 0)

        is_strong_made = rank.value >= HandRank.TWO_PAIR.value
        is_medium_made = rank == HandRank.PAIR
        is_draw = draw_type is not None and phase != GamePhase.RIVER
        is_garbage = not is_strong_made and not is_medium_made and not is_draw

        pot_odds_pct = self._calculate_pot_odds_percentage(pot, call_amount)
        draw_equity_pct = self._calculate_draw_equity_percentage(outs, phase)

        # --- Decision Logic ---
        if can_check:
            # RANDOM BETS when checked to
            if is_strong_made: # Value Bet
                amount = self._calculate_bet_amount(pot, big_blind, is_value=True, phase=phase)
                return PlayerAction.BET, amount
            elif is_medium_made and random.random() < self.POSTFLOP_RANDOM_BET_CHECKED_MEDIUM: # Random bet medium pair
                amount = self._calculate_bet_amount(pot, big_blind, is_value=False, phase=phase) # Bluff/probe sizing
                return PlayerAction.BET, amount
            elif is_draw and random.random() < self.POSTFLOP_RANDOM_BET_CHECKED_DRAW: # Random semi-bluff draw
                 amount = self._calculate_bet_amount(pot, big_blind, is_value=False, phase=phase)
                 return PlayerAction.BET, amount
            elif is_garbage and phase == GamePhase.FLOP and random.random() < self.POSTFLOP_RANDOM_BET_CHECKED_GARBAGE: # Random bluff garbage on flop
                 amount = self._calculate_bet_amount(pot, big_blind, is_value=False, phase=phase)
                 return PlayerAction.BET, amount
            else: # Default Check
                 return PlayerAction.CHECK, 0
        else: # Facing a bet
            # RANDOM RAISES when facing bet
            if is_strong_made: # Value Raise/Call
                 if rank.value >= HandRank.THREE_OF_A_KIND.value: # Raise Sets+
                       raise_amount = self._calculate_raise_amount(current_bet, big_blind, pot, num_players, is_value=True)
                       if raise_amount >= self.stack: return PlayerAction.ALL_IN, self.stack
                       if raise_amount > 0 : return PlayerAction.RAISE, raise_amount
                       else: return PlayerAction.CALL, call_amount # Fallback to call if raise calc fails
                 else: # Call Two Pair etc if affordable
                      if call_amount < self.stack * 0.35: return PlayerAction.CALL, call_amount
                      else: return PlayerAction.FOLD, 0
            elif is_medium_made and random.random() < self.POSTFLOP_RANDOM_RAISE_FACING_BET_MEDIUM: # Random bluff-raise medium pair
                 raise_amount = self._calculate_raise_amount(current_bet, big_blind, pot, num_players, is_value=False)
                 if raise_amount >= self.stack: return PlayerAction.ALL_IN, self.stack
                 if raise_amount > 0 : return PlayerAction.RAISE, raise_amount
                 else: return PlayerAction.FOLD, 0 # Fold if cannot bluff raise
            elif is_draw: # Draw logic (call w/ odds or random semi-bluff raise)
                if draw_equity_pct >= pot_odds_pct: # Call with odds
                    return PlayerAction.CALL, call_amount
                elif random.random() < self.POSTFLOP_RANDOM_RAISE_FACING_BET_DRAW: # Random semi-bluff raise draw
                     raise_amount = self._calculate_raise_amount(current_bet, big_blind, pot, num_players, is_value=False)
                     if raise_amount >= self.stack: return PlayerAction.ALL_IN, self.stack
                     if raise_amount > 0 : return PlayerAction.RAISE, raise_amount
                     else: return PlayerAction.FOLD, 0 # Fold if cannot semi-bluff raise
                else: # Fold draw without odds and no random raise
                     return PlayerAction.FOLD, 0
            else: # Garbage or Medium Pair (not randomly raised) - default fold/bluff catch
                 if is_medium_made and pot_odds_pct < 30 and call_amount < self.stack * 0.15: # Bluff catch small bets
                      return PlayerAction.CALL, call_amount
                 else: # Fold garbage or vs larger bets
                      return PlayerAction.FOLD, 0


# Note: _calculate_bet_size was renamed/split into _calculate_raise_amount and _calculate_bet_amount for clarity.
# Ensure these helper methods are correctly defined as above.
#juari