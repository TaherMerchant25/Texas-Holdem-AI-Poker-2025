import random
import math
from typing import List, Tuple, Optional, Dict
# Added deque import at the top level
from collections import deque

# Assuming card.py, player.py, hand_evaluator.py, game.py are accessible
# Ensure these modules are correctly implemented and available
from card import Card, Rank, Suit, Deck
from player import Player, PlayerAction # Base class must handle self.bet_amount correctly
from hand_evaluator import HandEvaluator, HandRank, HandResult # HandResult needs .hand_rank and ideally .cards
from game import GamePhase

# --- Helper function to get Card from index (assuming it exists and works) ---
_FULL_DECK_FOR_INDEXING = Deck().cards
def card_from_index(index: int) -> Optional[Card]:
    """Maps an index (0-51) back to a Card object."""
    if 0 <= index < 52:
        for card in _FULL_DECK_FOR_INDEXING:
             if card.get_index() == index:
                 return card
    return None
# --- End Helper ---

class AdaptiveRaisePlayer(Player):
    """
    An AI player that adapts its aggression and raise sizing based on
    hand strength, opponent modeling, and game context.
    It attempts to balance value betting and bluffing.

    ASSUMPTIONS:
    - Base Player class correctly manages self.bet_amount (updates per action, resets per betting round).
    - HandEvaluator returns a HandResult object with '.hand_rank' (HandRank enum) and ideally '.cards' (list of best 5 cards).
    - action_history provides tuples like (player_index, PlayerAction, amount) for the current hand.
    - card_from_index helper works as intended.
    """

    # --- Constants for Strategy ---
    # Sizing & Aggression (Using values from previous version)
    RAISE_SIZE_BASE_MULTIPLIER = 2.5
    BET_SIZE_POT_FRACTION_VALUE = 0.65
    BET_SIZE_POT_FRACTION_BLUFF = 0.50
    AGGRESSION_ADJUSTMENT_RATE = 0.07
    MIN_RAISE_AGGRESSION = 0.3
    MAX_RAISE_AGGRESSION = 1.5
    STARTING_AGGRESSION = 0.8

    # Bluffing & Hand Strength
    BLUFF_TO_VALUE_RATIO = 0.4
    GOOD_HAND_WIN_RATE = 0.65
    STRONG_HAND_WIN_RATE = 0.80
    PREFLOP_PREMIUM_THRESHOLD = 0.85
    PREFLOP_STRONG_THRESHOLD = 0.70
    PREFLOP_GOOD_THRESHOLD = 0.55

    # Opponent Modeling & Adaptation
    HISTORY_LENGTH = 20
    OPPONENT_AGGRESSION_THRESHOLD = 0.4
    OPPONENT_FOLD_THRESHOLD_HIGH = 0.6
    AGGRESSION_BOOST_VS_PASSIVE = 0.15
    AGGRESSION_REDUCE_VS_AGGRESSIVE = 0.10
    BLUFF_BOOST_VS_TIGHT = 0.1

    # Stack & Risk Management
    DEFENSIVE_STACK_THRESHOLD_BB = 20
    PREFLOP_ALLIN_THRESHOLD_BB = 15
    MAX_RISK_PERCENTAGE_BLUFF = 0.3
    MAX_RISK_PERCENTAGE_CALL = 0.4

    # --- Instance Variables ---
    def __init__(self, name: str, stack: int):
        super().__init__(name, stack)
        self.aggression = self.STARTING_AGGRESSION
        # Opponent models store: {player_index: {"actions": deque, "fold_rate": float, "aggression_rate": float, "is_aggressive": bool, "is_tight": bool}}
        self.opponent_models: Dict[int, Dict] = {}
        self.my_index = None # Will be set in the action method
        self.initial_stack = stack # Store initial stack if needed for analysis later
        # Game stats (optional)
        self.wins = 0
        self.total_games = 0
        self.winning_percentage = 0.0

        # Internal state for the current hand
        self._current_bb = 10 # Default BB if not provided initially (should be updated)
        self.hole_cards: List[Card] = [] # Initialize hole cards list

    # --- Main Action Method ---
    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
        """Determines the player's action based on game state, hand strength, and opponent models."""

        try: # Added try-except block for better error catching during execution
            # --- 1. Parse Game State ---
            hole_card_indices = game_state[0:2]
            community_card_indices = game_state[2:7]
            pot = game_state[7]
            current_bet = game_state[8] # The total amount players need to call
            # Ensure BB is valid
            self._current_bb = max(1, game_state[9]) # Store BB size, prevent division by zero
            self.my_index = game_state[10]
            num_players = game_state[11]
            # Ensure player_stacks slicing is correct based on num_players
            player_stacks = game_state[12 : 12 + num_players]

            # --- Get actual Card objects ---
            self.hole_cards = [card for i in hole_card_indices if (card := card_from_index(i))]
            community_cards = [card for i in community_card_indices if (card := card_from_index(i))]

            # Safety check for hole cards
            if not self.hole_cards or len(self.hole_cards) != 2 :
                 print(f"AI {self.name} received invalid hole cards: {hole_card_indices}. Folding.")
                 # Assuming self.bet_amount exists from base Player class
                 call_amount_safe = max(0, current_bet - self.bet_amount)
                 return (PlayerAction.CHECK, 0) if call_amount_safe == 0 else (PlayerAction.FOLD, 0)

            # --- 2. Update Internal State & Models ---
            # Check if action_history is usable before updating models
            if isinstance(action_history, list):
                self._update_opponent_models(action_history, num_players)
                self._adjust_aggression() # Adjust based on models
            else:
                print(f"AI {self.name}: Warning - Invalid action_history format received.")


            # --- 3. Calculate Hand Context ---
            phase = self._get_phase(community_cards)
            # CRITICAL: Assumes self.bet_amount is correctly managed by the base Player class
            # It should reflect the amount *this player* has put in *this betting round*
            call_amount = max(0, current_bet - self.bet_amount)
            can_check = (call_amount == 0)
            my_stack = self.stack # Convenience

            # --- 4. Determine Action Based on Phase ---
            if phase == GamePhase.PRE_FLOP:
                return self._decide_preflop(call_amount, current_bet, pot, num_players, player_stacks)
            else:
                # Ensure community cards are valid for postflop evaluation
                if phase != GamePhase.SETUP:
                     return self._decide_postflop(phase, community_cards, call_amount, current_bet, pot, num_players, player_stacks)
                else:
                     print(f"AI {self.name}: Invalid game phase detected post-flop. Folding.")
                     return (PlayerAction.CHECK, 0) if can_check else (PlayerAction.FOLD, 0)

        except IndexError as e:
             print(f"AI {self.name}: Error accessing game_state element: {e}. State: {game_state}. Folding.")
             # Fallback action on error
             call_amount_safe = max(0, game_state[8] - self.bet_amount) # Use index directly if state is bad
             return (PlayerAction.CHECK, 0) if call_amount_safe == 0 else (PlayerAction.FOLD, 0)
        except Exception as e: # Catch any other unexpected error
             print(f"AI {self.name}: Unexpected error in action method: {e}. Folding.")
             call_amount_safe = max(0, game_state[8] - self.bet_amount)
             return (PlayerAction.CHECK, 0) if call_amount_safe == 0 else (PlayerAction.FOLD, 0)


    # --- Helper Methods ---
    def _get_phase(self, community_cards: List[Card]) -> GamePhase:
        """Determines the current game phase."""
        num_community = len(community_cards)
        if num_community == 0: return GamePhase.PRE_FLOP
        elif num_community == 3: return GamePhase.FLOP
        elif num_community == 4: return GamePhase.TURN
        elif num_community == 5: return GamePhase.RIVER
        else: return GamePhase.SETUP # Indicates an issue or pre-deal state

    def _estimate_preflop_strength(self) -> float:
        """Estimates pre-flop hand strength as a value between 0 and 1."""
        if not self.hole_cards or len(self.hole_cards) != 2: return 0.0
        # --- Using the same heuristic logic as before ---
        c1, c2 = sorted(self.hole_cards, key=lambda c: c.rank.value, reverse=True)
        r1_val, r2_val = c1.rank.value, c2.rank.value
        suited = c1.suit == c2.suit
        is_pair = r1_val == r2_val
        gap = r1_val - r2_val - 1 if not is_pair else 0

        strength = 0.0
        strength += (r1_val + r2_val) / (Rank.ACE.value * 2) * 0.6
        if is_pair:
            strength += (r1_val / Rank.ACE.value) * 0.4
            if r1_val >= Rank.TEN.value: strength += 0.1
        if suited:
            strength += 0.15
        if not is_pair:
            if gap == 0: strength += 0.12 if suited else 0.06
            elif gap == 1: strength += 0.08 if suited else 0.03
            elif gap == 2: strength += 0.05 if suited else 0.01
        if r1_val == Rank.ACE.value: strength += 0.1
        return min(strength, 1.0)

    def _estimate_postflop_strength(self, community_cards: List[Card]) -> float:
        """Estimates post-flop hand strength (win rate approximation)."""
        if not self.hole_cards: return 0.0
        if not community_cards: return self._estimate_preflop_strength() # Should not happen if phase is post-flop

        hand_evaluator = HandEvaluator()
        try:
            # Ensure community_cards is a list of Card objects
            valid_community = [c for c in community_cards if isinstance(c, Card)]
            if len(valid_community) < 3: # Need at least flop
                 print(f"AI {self.name}: Warning - Not enough valid community cards for postflop eval. Len: {len(valid_community)}")
                 # Fallback to a low value, or preflop strength? Low value might be safer.
                 return 0.1
            hand_result = hand_evaluator.evaluate_hand(self.hole_cards, valid_community)
        except Exception as e:
            print(f"AI {self.name}: Error during hand evaluation: {e}. Hole: {self.hole_cards}, Comm: {community_cards}")
            return 0.0 # Return minimum strength on error

        rank = hand_result.hand_rank

        # --- Using the same rank-to-value mapping as before ---
        if rank == HandRank.ROYAL_FLUSH: return 1.0
        if rank == HandRank.STRAIGHT_FLUSH: return 0.98
        if rank == HandRank.FOUR_OF_A_KIND: return 0.95
        if rank == HandRank.FULL_HOUSE: return 0.88
        if rank == HandRank.FLUSH: return 0.78
        if rank == HandRank.STRAIGHT: return 0.70
        if rank == HandRank.THREE_OF_A_KIND: return 0.60
        if rank == HandRank.TWO_PAIR: return 0.50
        if rank == HandRank.PAIR:
            pair_rank_val = Rank.TWO.value # Default to lowest pair
            # Check if HandResult provides the cards forming the hand
            # This is a potential point of failure if the evaluator doesn't provide it
            if hasattr(hand_result, 'cards') and hand_result.cards:
                try:
                    ranks_in_hand = [c.rank.value for c in hand_result.cards]
                    rank_counts = {r: ranks_in_hand.count(r) for r in set(ranks_in_hand)}
                    for r, count in rank_counts.items():
                        if count == 2:
                            pair_rank_val = r
                            break
                except Exception as e:
                     print(f"AI {self.name}: Error processing hand_result.cards for pair: {e}")
                     # Keep default pair_rank_val
            else:
                 # If cards not available, we can't determine pair rank easily. Use a mid-value.
                 return 0.35 # Generic mid-pair value

            normalized_pair_rank = max(0, (pair_rank_val - Rank.TWO.value)) / max(1, (Rank.ACE.value - Rank.TWO.value))
            return 0.25 + (normalized_pair_rank * 0.20)
        if rank == HandRank.HIGH_CARD:
             try:
                 # Ensure cards is iterable and contains Card objects
                 all_cards = self.hole_cards + valid_community
                 if not all_cards: return 0.1 # Should not happen here
                 high_card_val = max(c.rank.value for c in all_cards if isinstance(c, Card))
                 normalized_high_card = max(0, (high_card_val - Rank.TWO.value)) / max(1, (Rank.ACE.value - Rank.TWO.value))
                 return 0.10 + (normalized_high_card * 0.15)
             except Exception as e:
                  print(f"AI {self.name}: Error processing high card: {e}")
                  return 0.1 # Fallback

        return 0.0 # Default for unknown rank

    # --- Decision Logic Methods (Largely unchanged logic, ensure variables used are defined) ---
    def _decide_preflop(self, call_amount: int, current_bet: int, pot: int, num_players: int, player_stacks: List[int]) -> Tuple[PlayerAction, int]:
        """Makes the pre-flop decision."""
        strength = self._estimate_preflop_strength()
        can_check = (call_amount == 0)
        my_stack_bb = self.stack / self._current_bb if self._current_bb > 0 else 0

        # --- Push/Fold Strategy ---
        if my_stack_bb <= self.PREFLOP_ALLIN_THRESHOLD_BB:
            push_threshold = self.PREFLOP_GOOD_THRESHOLD - ((self.PREFLOP_ALLIN_THRESHOLD_BB - my_stack_bb) * 0.02)
            if strength >= push_threshold:
                return PlayerAction.ALL_IN, self.stack
            else:
                if call_amount > 0 and call_amount >= self.stack: return PlayerAction.FOLD, 0
                if call_amount > 0: return PlayerAction.FOLD, 0
                else: return PlayerAction.CHECK, 0

        # --- Standard Pre-flop Play ---
        # Simple check if anyone raised before us (bet > BB)
        action_to_me = current_bet > self._current_bb

        # A) Opening Action
        if not action_to_me: # Check if effective bet is just BB or less
             # Check if we are effectively opening or just calling BB/limps
             is_opening = self.bet_amount == 0 or self.bet_amount == self._current_bb / 2 # Small blind case approx
             effective_current_bet = current_bet if current_bet > self._current_bb else self._current_bb # Treat limps/BB as the 'bet' to raise over

             if strength >= self.PREFLOP_STRONG_THRESHOLD:
                 target_bet = self._calculate_bet_amount(pot, is_value=True, phase=GamePhase.PRE_FLOP)
                 # Ensure raise is valid over effective bet
                 min_raise_total = effective_current_bet + max(self._current_bb, effective_current_bet) # Approx min raise logic
                 target_bet = max(target_bet, min_raise_total)
                 target_bet = min(target_bet, self.stack)
                 # Use RAISE if facing BB/limp, BET if first action
                 action = PlayerAction.RAISE if effective_current_bet > 0 else PlayerAction.BET
                 # Ensure we are actually increasing the bet amount significantly
                 if target_bet > current_bet and target_bet > self.bet_amount:
                      # Need to return the TOTAL bet amount for BET/RAISE
                      return action, target_bet
                 else: # Fallback if calculated bet isn't valid raise/bet
                      return PlayerAction.CALL, call_amount if call_amount > 0 else PlayerAction.CHECK, 0

             elif strength >= self.PREFLOP_GOOD_THRESHOLD:
                 if random.random() < (self.aggression * 0.5):
                     target_bet = self._calculate_bet_amount(pot, is_value=True, phase=GamePhase.PRE_FLOP)
                     min_raise_total = effective_current_bet + max(self._current_bb, effective_current_bet)
                     target_bet = max(target_bet, min_raise_total)
                     target_bet = min(target_bet, self.stack)
                     action = PlayerAction.RAISE if effective_current_bet > 0 else PlayerAction.BET
                     if target_bet > current_bet and target_bet > self.bet_amount:
                         return action, target_bet
                 # Default: Call if affordable
                 if call_amount < self.stack * self.MAX_RISK_PERCENTAGE_CALL and call_amount <= self.stack:
                     return PlayerAction.CALL, call_amount if call_amount > 0 else PlayerAction.CHECK, 0
                 else:
                     return PlayerAction.FOLD, 0
             else: # Weak hands
                 return PlayerAction.CHECK, 0 if can_check else PlayerAction.FOLD, 0

        # B) Facing a Raise
        else:
            if strength >= self.PREFLOP_PREMIUM_THRESHOLD:
                 # Calculate the *additional* amount for the raise
                 raise_amount_by = self._calculate_raise_amount(current_bet, pot, is_value=True, phase = GamePhase.PRE_FLOP)
                 if raise_amount_by > 0:
                      # Return the TOTAL bet amount
                      total_bet = self.bet_amount + raise_amount_by
                      return PlayerAction.RAISE, min(total_bet, self.stack) # Ensure not over stack
                 else: # If cannot make valid raise, call if possible
                      if call_amount < self.stack * self.MAX_RISK_PERCENTAGE_CALL and call_amount <= self.stack:
                           return PlayerAction.CALL, call_amount
                      else: return PlayerAction.FOLD, 0

            elif strength >= self.PREFLOP_STRONG_THRESHOLD:
                 if random.random() < (self.aggression * 0.4):
                       raise_amount_by = self._calculate_raise_amount(current_bet, pot, is_value=True, phase = GamePhase.PRE_FLOP)
                       if raise_amount_by > 0:
                            total_bet = self.bet_amount + raise_amount_by
                            return PlayerAction.RAISE, min(total_bet, self.stack)
                 # Default: Call if affordable
                 if call_amount < self.stack * self.MAX_RISK_PERCENTAGE_CALL and call_amount <= self.stack:
                       return PlayerAction.CALL, call_amount
                 else:
                       return PlayerAction.FOLD, 0

            elif strength >= self.PREFLOP_GOOD_THRESHOLD:
                 pot_odds_pct = (call_amount / (pot + call_amount + call_amount)) * 100 if (pot + 2 * call_amount) > 0 else 100
                 # Call small raises with good speculative hands if affordable
                 if call_amount < pot * 0.3 and call_amount < self.stack * (self.MAX_RISK_PERCENTAGE_CALL * 0.7) and call_amount <= self.stack:
                       return PlayerAction.CALL, call_amount
                 else:
                       return PlayerAction.FOLD, 0
            else: # Weak hands
                 return PlayerAction.FOLD, 0

    def _decide_postflop(self, phase: GamePhase, community_cards: List[Card], call_amount: int, current_bet: int, pot: int, num_players: int, player_stacks: List[int]) -> Tuple[PlayerAction, int]:
        """Makes the post-flop decision."""
        win_rate = self._estimate_postflop_strength(community_cards)
        can_check = (call_amount == 0)
        my_stack_bb = self.stack / self._current_bb if self._current_bb > 0 else 0
        play_defensively = my_stack_bb < self.DEFENSIVE_STACK_THRESHOLD_BB

        # A) Action is Checked to Us
        if can_check:
            is_value_hand = win_rate >= self.GOOD_HAND_WIN_RATE
            # Determine bluff chance, considering aggression and opponent tightness
            bluff_chance = self.aggression * self.BLUFF_TO_VALUE_RATIO
            tight_opponents = sum(1 for model in self.opponent_models.values() if model.get("is_tight", False))
            if tight_opponents > len(self.opponent_models) / 2 and len(self.opponent_models) > 0:
                 bluff_chance += self.BLUFF_BOOST_VS_TIGHT
            should_bluff = (random.random() < bluff_chance) and not is_value_hand

            if is_value_hand:
                 bet_amount = self._calculate_bet_amount(pot, is_value=True, phase=phase)
                 # Ensure bet amount is valid
                 bet_amount = max(self._current_bb, min(bet_amount, self.stack))
                 return PlayerAction.BET, bet_amount
            elif should_bluff and not play_defensively:
                 bet_amount = self._calculate_bet_amount(pot, is_value=False, phase=phase)
                 bet_amount = min(bet_amount, int(self.stack * self.MAX_RISK_PERCENTAGE_BLUFF))
                 bet_amount = max(self._current_bb, min(bet_amount, self.stack)) # Ensure min bet and cap
                 if bet_amount >= self._current_bb:
                     return PlayerAction.BET, bet_amount
                 else:
                     return PlayerAction.CHECK, 0 # Cannot make meaningful bluff
            else:
                 return PlayerAction.CHECK, 0

        # B) Facing a Bet
        else:
            pot_odds = call_amount / (pot + call_amount + call_amount) if (pot + 2 * call_amount) > 0 else 1.0 # Assign 1.0 if call_amount is 0 to avoid division by zero, though this case shouldn't happen if not can_check
            required_equity = pot_odds

            is_strong_value_hand = win_rate >= self.STRONG_HAND_WIN_RATE
            is_good_value_hand = win_rate >= self.GOOD_HAND_WIN_RATE

            if is_strong_value_hand:
                 raise_amount_by = self._calculate_raise_amount(current_bet, pot, is_value=True, phase=phase)
                 if raise_amount_by > 0:
                      total_bet = self.bet_amount + raise_amount_by
                      # Simple all-in check
                      if total_bet >= self.stack * 0.8 : # Go all-in if committing most of stack
                          return PlayerAction.ALL_IN, self.stack
                      else:
                          return PlayerAction.RAISE, min(total_bet, self.stack)
                 else: # Cannot raise, call if affordable
                      if call_amount <= self.stack: return PlayerAction.CALL, call_amount
                      else: return PlayerAction.FOLD, 0 # Should not happen if stack > 0

            elif is_good_value_hand:
                 if win_rate >= required_equity: # Check if getting odds
                      if random.random() < (self.aggression * 0.3) and not play_defensively:
                           raise_amount_by = self._calculate_raise_amount(current_bet, pot, is_value=True, phase=phase)
                           if raise_amount_by > 0:
                                total_bet = self.bet_amount + raise_amount_by
                                # Avoid massive over-raise unless necessary
                                total_bet = min(total_bet, int(self.stack * 0.7))
                                total_bet = max(total_bet, current_bet + max(self._current_bb, current_bet)) # Ensure minimum legal total
                                if total_bet > current_bet: # Ensure it's a raise
                                     return PlayerAction.RAISE, min(total_bet, self.stack)
                      # Default: Call if affordable and meets risk threshold
                      if call_amount < self.stack * self.MAX_RISK_PERCENTAGE_CALL and call_amount <= self.stack:
                           return PlayerAction.CALL, call_amount
                      else:
                           return PlayerAction.FOLD, 0 # Cannot afford call risk
                 else:
                       return PlayerAction.FOLD, 0 # Not getting odds

            else: # Weak Hand / Bluff Catcher / Draw (win_rate low)
                 bluff_raise_chance = self.aggression * self.BLUFF_TO_VALUE_RATIO * 0.5
                 foldy_opponent = any(m.get("is_tight", False) for m in self.opponent_models.values())
                 should_bluff_raise = (random.random() < bluff_raise_chance) and foldy_opponent

                 if should_bluff_raise and not play_defensively:
                       raise_amount_by = self._calculate_raise_amount(current_bet, pot, is_value=False, phase=phase)
                       if raise_amount_by > 0:
                            total_bet = self.bet_amount + raise_amount_by
                            # Limit total risk on bluff raise
                            if total_bet <= self.stack * self.MAX_RISK_PERCENTAGE_BLUFF and total_bet <= self.stack:
                                 return PlayerAction.RAISE, total_bet
                            # Else fall through if bluff raise is too risky

                 # Cheap bluff catch
                 is_cheap_bluff_catch = required_equity < 0.20 and call_amount < self.stack * 0.10
                 if is_cheap_bluff_catch and not play_defensively and call_amount <= self.stack:
                      return PlayerAction.CALL, call_amount

                 # Default: Fold
                 return PlayerAction.FOLD, 0

    # --- Bet/Raise Sizing ---
    def _calculate_bet_amount(self, pot: int, is_value: bool, phase: GamePhase) -> int:
        """Calculates the TOTAL amount TO BET (opening action post-flop or pre-flop)."""
        pot_fraction = self.BET_SIZE_POT_FRACTION_VALUE if is_value else self.BET_SIZE_POT_FRACTION_BLUFF
        aggression_multiplier = 1 + (self.aggression - self.STARTING_AGGRESSION) * 0.5
        pot_fraction *= aggression_multiplier
        pot_fraction = max(0.1, min(pot_fraction, 1.5)) # Clamp pot fraction

        if phase == GamePhase.PRE_FLOP:
             base_size = self._current_bb * 3
             size = int(base_size * (1 + (self.aggression - self.STARTING_AGGRESSION) * 0.2))
        else:
             size = int(pot * pot_fraction)

        min_bet = self._current_bb
        # Ensure bet is at least min bet, and capped reasonably (e.g., pot size, or stack)
        size = max(min_bet, size)
        # Let action method cap at stack if needed
        return size

    def _calculate_raise_amount(self, current_bet: int, pot: int, is_value: bool, phase: GamePhase) -> int:
        """
        Calculates the amount TO RAISE BY (additional amount).
        Returns 0 if raise is not possible/sensible.
        """
        # Ensure current_bet is positive, otherwise it's not a raise scenario
        if current_bet <= 0: return 0

        # Use self.bet_amount from base Player class - amount already invested this round
        call_amount = max(0, current_bet - self.bet_amount)

        # Minimum legal increase: Last bet/raise size. Approximated by current_bet size
        # or BB if current_bet is just the BB. More complex history parsing needed for perfect accuracy.
        last_bet_or_raise_size = current_bet - (pot - current_bet) # Rough estimate - needs history
        last_bet_or_raise_size = max(self._current_bb, last_bet_or_raise_size if last_bet_or_raise_size > 0 else self._current_bb) # Fallback to BB
        min_legal_increase = last_bet_or_raise_size # Use this approximation

        base_multiplier = self.RAISE_SIZE_BASE_MULTIPLIER
        aggression_multiplier = 1 + (self.aggression - self.STARTING_AGGRESSION) * 0.4
        effective_multiplier = base_multiplier * aggression_multiplier

        # Target *total* bet after our raise
        # Raising to X times the *previous* bet/raise size is common
        # target_total_bet = current_bet + (min_legal_increase * effective_multiplier) # Option 1: Raise X * increase
        target_total_bet = int(current_bet * effective_multiplier) # Option 2: Raise to X * current total bet (simpler)


        if is_value:
             target_pot_size_raise = int((pot + current_bet + call_amount) * 0.75)
             target_total_bet = max(target_total_bet, target_pot_size_raise)

        min_legal_total_bet = current_bet + min_legal_increase
        target_total_bet = max(target_total_bet, min_legal_total_bet)

        # Calculate the actual amount *to raise by* (additional chips needed from us)
        raise_amount_by = target_total_bet - self.bet_amount
        raise_amount_by = max(0, raise_amount_by) # Cannot be negative

        # Check affordability vs stack
        if raise_amount_by > self.stack:
             raise_amount_by = self.stack # Capped at stack (leads to all-in)

        # Raise must be strictly greater than just calling
        if raise_amount_by <= call_amount:
             # Try forcing minimum legal raise if possible
             min_legal_raise_amount_by = min_legal_total_bet - self.bet_amount
             min_legal_raise_amount_by = max(0, min_legal_raise_amount_by)

             if min_legal_raise_amount_by > call_amount and min_legal_raise_amount_by <= self.stack:
                  raise_amount_by = min_legal_raise_amount_by
             else:
                  return 0 # Cannot make a valid raise > call

        # Cap bluff raise risk (based on TOTAL commitment)
        if not is_value:
            total_bluff_commitment = self.bet_amount + raise_amount_by
            max_bluff_risk = int(self.stack * self.MAX_RISK_PERCENTAGE_BLUFF)
            if total_bluff_commitment > max_bluff_risk:
                 # Reduce raise amount to meet max risk
                 allowed_raise_by = max_bluff_risk - self.bet_amount
                 if allowed_raise_by > call_amount and allowed_raise_by > 0:
                     raise_amount_by = allowed_raise_by
                 else:
                     return 0 # Cannot make a valid bluff raise within risk limits

        # Final check: ensure the calculated amount is possible and > call
        if raise_amount_by <= call_amount or raise_amount_by <= 0:
            return 0

        return int(raise_amount_by)


    # --- Opponent Modeling & Aggression Adjustment ---
    def _update_opponent_models(self, action_history: list, num_players: int) -> None:
        """Updates models of opponents based on their recent actions."""
        # Initialize models if they don't exist
        for i in range(num_players):
            if i == self.my_index:
                continue
            if i not in self.opponent_models:
                self.opponent_models[i] = {
                    "actions": deque(maxlen=self.HISTORY_LENGTH), # Use deque here
                    "fold_rate": 0.0, "aggression_rate": 0.0,
                    "is_aggressive": False, "is_tight": False
                }

        # Process history - ASSUMES format (player_idx, PlayerAction, amount)
        processed_indices_this_round = set() # Avoid double counting if history spans rounds incorrectly
        for item in reversed(action_history): # Process recent actions first
            if isinstance(item, tuple) and len(item) >= 2:
                 p_idx, action = item[0], item[1]
                 # Check if player is an opponent and not already processed in this update cycle if needed
                 if p_idx != self.my_index and p_idx in self.opponent_models: # and p_idx not in processed_indices_this_round:
                       self.opponent_models[p_idx]["actions"].append(action)
                       # processed_indices_this_round.add(p_idx) # Uncomment if history might contain duplicates per round
            # else: print warning handled in action method

        # Recalculate stats
        for i in self.opponent_models:
            actions = self.opponent_models[i]["actions"]
            total_actions = len(actions)
            if total_actions > 0:
                fold_count = sum(1 for a in actions if a == PlayerAction.FOLD)
                aggressive_actions = {PlayerAction.BET, PlayerAction.RAISE, PlayerAction.ALL_IN}
                aggressive_count = sum(1 for a in actions if a in aggressive_actions)

                self.opponent_models[i]["fold_rate"] = fold_count / total_actions
                self.opponent_models[i]["aggression_rate"] = aggressive_count / total_actions
                self.opponent_models[i]["is_aggressive"] = self.opponent_models[i]["aggression_rate"] > self.OPPONENT_AGGRESSION_THRESHOLD
                self.opponent_models[i]["is_tight"] = self.opponent_models[i]["fold_rate"] > self.OPPONENT_FOLD_THRESHOLD_HIGH

    def _adjust_aggression(self) -> None:
        """Adjusts the AI's aggression based on opponent tendencies."""
        num_opponents = len(self.opponent_models)
        if num_opponents == 0: return

        # Ensure models are populated before using them
        valid_models = [m for m in self.opponent_models.values() if len(m.get("actions", [])) > 0]
        num_valid_models = len(valid_models)
        if num_valid_models == 0: return

        passive_opponents = sum(1 for model in valid_models if not model["is_aggressive"])
        aggressive_opponents = sum(1 for model in valid_models if model["is_aggressive"])

        adjustment = 0
        if passive_opponents / num_valid_models > 0.6:
             adjustment += self.AGGRESSION_BOOST_VS_PASSIVE
        elif aggressive_opponents / num_valid_models > 0.5:
             adjustment -= self.AGGRESSION_REDUCE_VS_AGGRESSIVE

        self.aggression += adjustment
        self.aggression = max(self.MIN_RAISE_AGGRESSION, min(self.aggression, self.MAX_RAISE_AGGRESSION))


    # --- Game Tracking Methods (Optional) ---
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
        # self.opponent_models = {} # Optionally reset models too
        # self.aggression = self.STARTING_AGGRESSION