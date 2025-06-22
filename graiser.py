import random
import math
from typing import List, Tuple, Optional, Set

# --- (Imports and Stub Definitions - Keep As Is) ---
try:
    from card import Card, Rank, Suit, Deck
    from player import Player, PlayerAction
    from hand_evaluator import HandEvaluator, HandRank
except ImportError:
    # Define minimal stubs if files are missing, for basic code analysis
    print("Warning: Could not import poker classes. Using minimal stubs.")
    class Card:
        def __init__(self, rank, suit): self.rank = rank; self.suit = suit
        def get_index(self): return random.randint(0,51) # Dummy index
        def __str__(self): return f"{self.rank}{self.suit}"
        @property
        def value(self): # Add dummy value if needed by evaluator/logic
            # Simple mapping, adjust if Rank enum is different
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            return rank_map.get(str(self.rank), 0)

    class Rank:
        TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = '2','3','4','5','6','7','8','9','T','J','Q','K','A'
        # Add value attribute if your evaluator uses it directly
        # Or ensure Card class provides it as above
        @property
        def value(self):
             rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
             return rank_map.get(str(self), 0)

    class Suit:
        SPADES, HEARTS, DIAMONDS, CLUBS = 'S', 'H', 'D', 'C'
        # Make Suit iterable for draw checks
        def __iter__(self):
            yield Suit.SPADES
            yield Suit.HEARTS
            yield Suit.DIAMONDS
            yield Suit.CLUBS

    class Deck:
        def __init__(self):
            # Create a dummy deck for card_from_index fallback
            self.cards = []
            try:
                ranks = [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
                suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
                self.cards = [Card(r, s) for s in suits for r in ranks]
                # Assign dummy indices
                for i, card in enumerate(self.cards):
                    card._index = i # Use a private attribute for dummy index
                    def get_dummy_index(self): return self._index
                    card.get_index = get_dummy_index.__get__(card, Card) # Bind method
            except Exception as e:
                 print(f"Warning: Failed to create dummy deck: {e}")
                 self.cards = []

    class PlayerAction: FOLD, CHECK, CALL, BET, RAISE, ALL_IN = range(6)
    class Player:
        def __init__(self, name: str, stack: int): self.name = name; self.stack = stack; self.hole_cards = []
        def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]: pass # Abstract
        # Add dummy hole_cards attribute if needed by evaluator
        def set_hole_cards(self, cards): self.hole_cards = cards

    class HandEvaluator:
        def evaluate_hand(self, hole_cards, community_cards):
            # print(f"Dummy Evaluating: {hole_cards} {community_cards}")
            class DummyResult:
                class DummyRank: name = "HIGH_CARD"; value=1 # Use value 1 for HIGH_CARD
                hand_rank = DummyRank()
                best_hand = hole_cards + community_cards[:5] # Dummy best hand
            # Very basic pair check for slightly better dummy eval
            all_cards = hole_cards + community_cards
            ranks = [c.rank for c in all_cards]
            rank_counts = {r: ranks.count(r) for r in set(ranks)}
            if 4 in rank_counts.values(): DummyResult.hand_rank.name="FOUR_OF_A_KIND"; DummyResult.hand_rank.value=8
            elif 3 in rank_counts.values() and 2 in rank_counts.values(): DummyResult.hand_rank.name="FULL_HOUSE"; DummyResult.hand_rank.value=7
            # Add other simple checks if needed (Flush, Straight are harder without proper logic)
            elif 3 in rank_counts.values(): DummyResult.hand_rank.name="THREE_OF_A_KIND"; DummyResult.hand_rank.value=4
            elif list(rank_counts.values()).count(2) >= 2: DummyResult.hand_rank.name="TWO_PAIR"; DummyResult.hand_rank.value=3
            elif 2 in rank_counts.values(): DummyResult.hand_rank.name="PAIR"; DummyResult.hand_rank.value=2

            return DummyResult() # Dummy

    class HandRank: # Dummy Enum matching values used
        ROYAL_FLUSH = 10; STRAIGHT_FLUSH = 9; FOUR_OF_A_KIND = 8; FULL_HOUSE = 7
        FLUSH = 6; STRAIGHT = 5; THREE_OF_A_KIND = 4; TWO_PAIR = 3; PAIR = 2; HIGH_CARD = 1


# --- (card_from_index function - Keep As Is, ensure it works with stubs if needed) ---
_FULL_DECK_FOR_INDEXING = Deck().cards if Deck().cards else [] # Handle case where Deck might fail in stub
def card_from_index(index: int) -> Optional[Card]:
    """Retrieves a Card object from its index (0-51)."""
    if not _FULL_DECK_FOR_INDEXING: # Fallback for stub
        # Approximate mapping - requires Card/Rank/Suit stubs to be usable
        if 0 <= index < 52:
            rank_val_num = (index % 13) + 2 # 2-14
            suit_val_char = ['S', 'H', 'D', 'C'][index // 13]
            try:
                 # Map number back to Rank stub character/object
                 rank_map_num_to_char = {2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K', 14:'A'}
                 rank_char = rank_map_num_to_char.get(rank_val_num)
                 # Get Rank object/char - depends on stub definition
                 rank_obj = getattr(Rank, rank_char, rank_char) # Get matching Rank attribute, fallback to char
                 # Get Suit object/char
                 suit_obj = {'S': Suit.SPADES, 'H': Suit.HEARTS, 'D': Suit.DIAMONDS, 'C': Suit.CLUBS}.get(suit_val_char)

                 if rank_obj and suit_obj:
                      card = Card(rank_obj, suit_obj)
                      # Assign the index back if possible for consistency
                      card._index = index
                      def get_dummy_index(self): return self._index
                      card.get_index = get_dummy_index.__get__(card, Card)
                      return card
                 else: return None
            except Exception as e:
                 # print(f"Stub card_from_index Error: {e}") # Debugging
                 return None # Cannot create card
        return None

    # Original logic if deck exists
    # Optimization: If Deck() populates _FULL_DECK_FOR_INDEXING and indices match list index
    if 0 <= index < len(_FULL_DECK_FOR_INDEXING):
        card = _FULL_DECK_FOR_INDEXING[index]
        # Verify index if get_index method is reliable
        try:
            if card.get_index() == index:
                return card
        except Exception: # If get_index fails or card is bad
             pass # Fallback to linear search

    # Fallback linear search if direct access failed or indices are not sequential
    for card in _FULL_DECK_FOR_INDEXING:
        try: # Add try-except around get_index()
             if card.get_index() == index:
                 return card
        except Exception:
            continue # Skip card if get_index fails
    return None
# --- (AdaptiveRaisePlayer Class Definition) ---
class AdaptiveRaisePlayer(Player):
    """
    A poker player AI that always raises (or bets/all-ins), adapting aggression based
    on opponent models and hand strength, including draw potential.
    Includes fixes for potential invalid command errors.
    """

    # --- Constants for Tuning ---
    RAISE_SIZE_MULTIPLIER = 1.1       # Multiplier for basic raise size relative to pot
    AGGRESSION_ADJUSTMENT_RATE = 0.07 # How much aggression changes per adjustment cycle
    MIN_RAISE_AGGRESSION = 0.4        # Minimum aggression level
    STARTING_AGGRESSION = 0.8         # Initial aggression level
    BLUFF_TO_VALUE_RATIO = 0.5        # Target ratio of bluffs to value bets (not explicitly used in this simple version)

    HISTORY_LENGTH = 15               # Number of opponent actions to remember
    AGGRESSION_THRESHOLD = 0.5        # Ratio of aggressive actions to classify opponent as aggressive
    RAISEPLAYER_DETECTED_AGGRESSION_BOOST = 0.3 # Extra aggression if facing aggressive players

    DEFENSIVE_STACK_THRESHOLD_MULT = 1.5 # Multiplier of current bet to consider stack "at risk"

    # Hand Strength / Win Rate Thresholds
    GOOD_HAND_WIN_RATE = 0.7          # Threshold for considering a hand "good"
    PREFLOP_ALLIN_THRESHOLD = 0.95    # Win rate threshold to shove pre-flop (if implemented)
    ALLIN_DEVIATION_THRESHOLD = 0.95  # Very high win rate where AI might shove unpredictably
    MAX_RAISE_THRESHOLD_MULT = 0.5    # Max raise *delta* as a fraction of the pot (to prevent overly huge raises) - ADJUSTED MEANING

    # Draw evaluation constants
    FLUSH_DRAW_BONUS = 0.15           # Win rate bonus for having a flush draw
    OESD_BONUS = 0.12                 # Win rate bonus for an open-ended straight draw
    GUTSHOT_BONUS = 0.06              # Win rate bonus for a gutshot straight draw

    # --- Initialization ---
    def __init__(self, name: str, stack: int):
        super().__init__(name, stack)
        self.aggression = self.STARTING_AGGRESSION
        self.opponent_models = {}       # Stores data about opponents {player_index: model_dict}
        self.my_index = None            # My position at the table in the current hand
        self.wins = 0                   # Tournament win tracking
        self.total_games = 0            # Tournament game tracking
        self.winning_percentage = 0.0   # Tournament winning percentage
        self.initial_stack = stack      # Store initial stack for reference

    # --- Core Action Logic ---
    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
        """
        Determines the player's action based on game state, hand strength, and opponent models.
        Corrected logic to avoid invalid commands.
        """
        # --- Parse Game State ---
        # Indices based on the provided game state format description
        hole_card_indices = game_state[0:2]
        community_card_indices = game_state[2:7] # Up to 5 community cards, -1 if not dealt
        pot = game_state[7]
        current_bet_to_call = game_state[8] # The current bet amount TO CALL
        big_blind = game_state[9]
        self.my_index = game_state[10]
        num_players = game_state[11]
        player_stacks = game_state[12:(12 + num_players)] # Stack sizes for all players

        # --- Reliably Update Stack ---
        my_stack = 0
        if self.my_index is not None and 0 <= self.my_index < num_players:
             my_stack = player_stacks[self.my_index]
             self.stack = my_stack # Update internal stack representation
        else:
             # Fallback or error - use internal, but log warning
             my_stack = self.stack
             print(f"Warning: {self.name} could not determine player index {self.my_index} or stack from game_state. Using internal value: {my_stack}")
             if self.my_index is None: # Try to find index if missing (e.g. if game assigns it later)
                 # This is risky, depends on game engine logic
                 pass # Cannot reliably determine index here

        # --- Handle Zero Stack ---
        if my_stack <= 0:
            # If stack is 0, the only valid action is CHECK (or FOLD if facing bet, but engine handles this)
            # print(f"Debug: {self.name} Stack is {my_stack}, performing CHECK (or implied fold)")
            return PlayerAction.CHECK, 0

        # --- Update Models ---
        self._update_opponent_models(action_history, num_players)
        self._adjust_aggression(self.opponent_models) # Adjust aggression based on opponents

        # --- Evaluate Hand ---
        win_rate = self._estimate_hand_strength(game_state, action_history)

        # --- Decision Making ---

        # Determine Minimum Legal Raise Amount (more robustly)
        # This still requires knowledge of the *last raise size*, which isn't in game_state.
        # Common rule: Min raise delta = size of last bet/raise delta.
        # Simplification/Workaround: Assume min raise delta is at least the Big Blind.
        # The total bet must be at least current_bet_to_call + max(big_blind, last_raise_delta)
        # Let's enforce that the total raise is at least call + BB.
        min_legal_total_raise = current_bet_to_call + big_blind

        # 1. Unpredictable All-in with monster hands
        if win_rate > self.ALLIN_DEVIATION_THRESHOLD and random.random() > 0.05:
            # print(f"Debug: {self.name} Deviating with All-In (Win Rate: {win_rate:.2f}, Stack: {my_stack})")
            return PlayerAction.ALL_IN, my_stack

        # 2. Can potentially Raise or Bet? (stack > amount to call)
        if my_stack > current_bet_to_call:
            # Calculate desired *total* bet amount using the helper function
            # Pass current_bet_to_call, which is needed for context inside the helper
            desired_total_bet = self._calculate_total_bet_amount(current_bet_to_call, big_blind, pot, win_rate, my_stack)

            # Is this a BET (no current bet to call) or a RAISE?
            is_betting_round_open = (current_bet_to_call == 0)

            # Ensure the desired bet/raise is legal
            if is_betting_round_open:
                # This is a BET. Minimum bet is usually Big Blind.
                min_legal_bet = max(big_blind, 1) # Ensure bet is at least 1 chip or BB
                # Clamp desired bet to be at least the minimum bet and capped by stack
                final_total_bet = max(min_legal_bet, desired_total_bet)
                final_total_bet = min(final_total_bet, my_stack)
                # print(f"Debug: {self.name} Betting. Desired: {desired_total_bet}, Min Legal: {min_legal_bet}, Final: {final_total_bet}, Stack: {my_stack}")
                return PlayerAction.BET, final_total_bet # Use BET action
            else:
                # This is a RAISE. Minimum total raise is call amount + BB (our simplification)
                # Clamp desired raise to be at least the minimum legal raise and capped by stack
                final_total_bet = max(min_legal_total_raise, desired_total_bet)
                final_total_bet = min(final_total_bet, my_stack)

                # If the calculated final_total_bet is only enough to call, just call (engine might require this)
                # Or if it equals stack, it's All-in. This bot *always* wants to raise, so it should only call if forced All-in?
                # Let's stick to RAISE if > call amount, otherwise consider ALL_IN if == stack.
                if final_total_bet > current_bet_to_call:
                    # print(f"Debug: {self.name} Raising. Desired: {desired_total_bet}, Min Legal Total: {min_legal_total_raise}, Final: {final_total_bet}, To Call: {current_bet_to_call}, Stack: {my_stack}")
                    return PlayerAction.RAISE, final_total_bet
                elif final_total_bet == my_stack: # If raise calc results in stack size (maybe min raise forces it)
                     # print(f"Debug: {self.name} Raising (All-in). Desired: {desired_total_bet}, Min Legal Total: {min_legal_total_raise}, Final: {final_total_bet}, To Call: {current_bet_to_call}, Stack: {my_stack}")
                     return PlayerAction.ALL_IN, my_stack
                else: # Should not happen if min_legal_total_raise > current_bet_to_call, but safety check
                     # This implies calculated raise is <= call amount. Forced to call/fold/all-in.
                     # Since bot never folds, go all-in if call equals stack, otherwise something is wrong.
                     if current_bet_to_call >= my_stack:
                          # print(f"Debug: {self.name} Cannot Raise legally, forced All-In Call. To Call: {current_bet_to_call}, Stack: {my_stack}")
                          return PlayerAction.ALL_IN, my_stack
                     else:
                          # This state is weird - can afford call, but min raise is > stack? Or calc failed?
                          # Fallback: Call if possible, otherwise All-in. But bot should raise...
                          # Default to All-In to maintain aggressive posture if raise calc failed low.
                          print(f"Warning: {self.name} Raise calculation resulted in non-raise amount ({final_total_bet}). Defaulting to All-In. To Call: {current_bet_to_call}, Stack: {my_stack}")
                          return PlayerAction.ALL_IN, my_stack


        # 3. Cannot Raise/Bet (stack <= current_bet_to_call)
        else: # my_stack <= current_bet_to_call
             # Must Fold, Call (if possible), or go All-In. Bot never folds.
             # If calling costs the entire stack, it's an All-In call.
             # print(f"Debug: {self.name} Cannot Raise/Bet. Stack: {my_stack}, To Call: {current_bet_to_call}. Going All-In.")
             return PlayerAction.ALL_IN, my_stack


    # --- Hand Strength Evaluation (Keep As Is) ---
    def _estimate_hand_strength(self, game_state: list[int], action_history: list) -> float:
        """
        Estimates the probability of winning the hand based on current cards
        and potential draws. Includes check for straight draws.
        (Ensure Card, Rank, Suit stubs/imports are compatible)
        """
        hole_card_indices = game_state[0:2]
        community_card_indices = game_state[2:7]

        hole_cards = [card_from_index(idx) for idx in hole_card_indices if idx != -1]
        community_cards = [card_from_index(idx) for idx in community_card_indices if idx != -1]

        # Filter out None cards more carefully
        hole_cards = [c for c in hole_cards if c is not None]
        community_cards = [c for c in community_cards if c is not None]

        if not hole_cards or len(hole_cards) != 2:
            # print("Debug: Invalid hole cards for estimation.")
            return 0.1 # Very low probability if cards are invalid/missing

        current_hand = hole_cards + community_cards
        num_community = len(community_cards)

        # Basic evaluation using HandEvaluator
        hand_evaluator = HandEvaluator()
        base_win_rate = 0.1 # Default low value
        current_rank_value = 0
        current_rank_name = "ERROR"

        try:
            # Ensure cards passed are Card objects with necessary attributes
            eval_hole = [c for c in hole_cards if hasattr(c, 'rank') and hasattr(c, 'suit')]
            eval_comm = [c for c in community_cards if hasattr(c, 'rank') and hasattr(c, 'suit')]

            if len(eval_hole) == 2: # Need 2 valid hole cards
                hand_result = hand_evaluator.evaluate_hand(eval_hole, eval_comm)
                if hasattr(hand_result, 'hand_rank') and hasattr(hand_result.hand_rank, 'value') and hasattr(hand_result.hand_rank, 'name'):
                    current_rank_value = hand_result.hand_rank.value
                    current_rank_name = hand_result.hand_rank.name
                else:
                     print("Warning: Hand evaluation result malformed.")

                # --- Base Win Rate based on Current Hand Rank ---
                if current_rank_name == 'ROYAL_FLUSH': base_win_rate = 1.0
                elif current_rank_name == 'STRAIGHT_FLUSH': base_win_rate = 0.98
                elif current_rank_name == 'FOUR_OF_A_KIND': base_win_rate = 0.95
                elif current_rank_name == 'FULL_HOUSE': base_win_rate = 0.85
                elif current_rank_name == 'FLUSH': base_win_rate = 0.75
                elif current_rank_name == 'STRAIGHT': base_win_rate = 0.65
                elif current_rank_name == 'THREE_OF_A_KIND': base_win_rate = 0.55
                elif current_rank_name == 'TWO_PAIR': base_win_rate = 0.45
                elif current_rank_name == 'PAIR': base_win_rate = 0.35
                elif current_rank_name == 'HIGH_CARD':
                    # Check high card value (requires rank comparison/value)
                    try:
                         ranks_in_hand = sorted([c.rank.value for c in current_hand if hasattr(c, 'rank') and hasattr(c.rank, 'value')], reverse=True)
                         ace_val = getattr(Rank, 'ACE', None)
                         king_val = getattr(Rank, 'KING', None)
                         jack_val = getattr(Rank, 'JACK', None)

                         if ranks_in_hand and king_val is not None and hasattr(king_val, 'value') and ranks_in_hand[0] >= king_val.value:
                             base_win_rate = 0.25
                         elif ranks_in_hand and jack_val is not None and hasattr(jack_val, 'value') and ranks_in_hand[0] >= jack_val.value:
                             base_win_rate = 0.20
                         else: base_win_rate = 0.15
                    except Exception as e_rank:
                        # print(f"Debug: Error comparing high card ranks: {e_rank}")
                        base_win_rate = 0.15 # Fallback if rank comparison fails
                else: base_win_rate = 0.10 # Error or unknown
            else:
                 # print("Debug: Not enough valid hole cards for eval.")
                 base_win_rate = 0.1

        except Exception as e:
            print(f"Error during hand evaluation: {e}")
            # Fallback values assigned above

        # --- Adjust Win Rate Based on Draws (if not on river) ---
        draw_bonus = 0.0
        if num_community < 5: # Only check draws if river isn't dealt
            try:
                # Check for Flush Draw
                suits_in_hand = [c.suit for c in current_hand if hasattr(c, 'suit')]
                if suits_in_hand:
                     suit_counts = {s: suits_in_hand.count(s) for s in set(suits_in_hand)}
                     if 4 in suit_counts.values():
                          draw_bonus = max(draw_bonus, self.FLUSH_DRAW_BONUS)

                # Check for Straight Draw Potential
                straight_draw_type = self._check_straight_draw_potential(current_hand)
                if straight_draw_type == "OESD":
                    draw_bonus = max(draw_bonus, self.OESD_BONUS)
                elif straight_draw_type == "GUTSHOT":
                    draw_bonus = max(draw_bonus, self.GUTSHOT_BONUS)

            except Exception as e_draw:
                 print(f"Error during draw checking: {e_draw}")


        # Apply draw bonus (only if hand isn't already very strong)
        final_win_rate = base_win_rate
        try:
            straight_rank_val = getattr(HandRank, 'STRAIGHT', 5) # Default value if stub fails
            if current_rank_value < straight_rank_val:
                 final_win_rate = min(base_win_rate + draw_bonus, 0.99) # Cap win rate below 1.0
        except Exception as e_hr:
             print(f"Warning: Could not compare hand rank for draw bonus: {e_hr}")
             final_win_rate = min(base_win_rate + draw_bonus, 0.99) # Apply bonus anyway if check fails


        # --- Optional: Stage-based Adjustment ---
        if num_community == 3 and draw_bonus > 0:
            final_win_rate *= 1.05
        elif num_community == 4 and draw_bonus > 0:
            final_win_rate *= 1.02

        # --- Final Cap ---
        final_win_rate = min(final_win_rate, 0.99) # Ensure it never reaches 1.0 unless Royal Flush

        # print(f"Debug Hand Eval: {[str(c) for c in hole_cards]} | {[str(c) for c in community_cards]} -> Rank: {current_rank_name}, Value: {current_rank_value}, BaseRate: {base_win_rate:.2f}, DrawBonus: {draw_bonus:.2f}, FinalRate: {final_win_rate:.2f}")
        return final_win_rate


    # --- Helper Function for Straight Draws (Keep As Is) ---
    def _check_straight_draw_potential(self, cards: List[Card]) -> Optional[str]:
        """
        Checks if the current hand contains a straight draw.
        Returns "OESD" (Open-Ended Straight Draw), "GUTSHOT", or None.
        (Requires Card objects have rank with value attribute, e.g., 2-14)
        """
        if not cards or len(cards) < 4: # Need at least 4 cards for a draw
            return None

        rank_values = set()
        has_ace = False
        try:
            for c in cards:
                 if hasattr(c, 'rank') and hasattr(c.rank, 'value'):
                      rank_val = c.rank.value
                      rank_values.add(rank_val)
                      if rank_val == getattr(Rank, 'ACE', type('obj', (object,), {'value': 14})()).value: # Check Ace value (default 14)
                           has_ace = True
                 else:
                      # print(f"Debug: Card {c} missing rank/value for straight check.")
                      pass # Skip card if unusable

            if not rank_values: return None # No valid ranks

            if has_ace:
                 rank_values.add(1) # Ace low value (1)

            unique_ranks = sorted(list(rank_values))
            if len(unique_ranks) < 4: return None # Not enough unique ranks for a draw

            # --- Check for OESD (4 consecutive ranks) ---
            for i in range(len(unique_ranks) - 3):
                # Check for 4 consecutive (handle potential non-integer ranks gracefully if needed)
                try:
                     if all(unique_ranks[i+j] == unique_ranks[i] + j for j in range(1, 4)):
                         # 4 consecutive found (e.g., unique_ranks[i] to unique_ranks[i]+3)
                         is_made_straight = False
                         # Check if 5 consecutive exist already
                         if i > 0 and unique_ranks[i-1] == unique_ranks[i] - 1: is_made_straight = True
                         if i + 4 < len(unique_ranks) and unique_ranks[i+4] == unique_ranks[i+3] + 1: is_made_straight = True
                         # Check for Ace-5 straight (if sequence is A,2,3,4 or 2,3,4,5)
                         ace_val = getattr(Rank, 'ACE', type('obj', (object,), {'value': 14})()).value
                         if unique_ranks[i:i+4] == [1, 2, 3, 4] and ace_val in rank_values: is_made_straight = True # A,2,3,4 + high A -> Made 5-high
                         if unique_ranks[i:i+4] == [2, 3, 4, 5] and 1 in rank_values: is_made_straight = True # 2,3,4,5 + low A -> Made 5-high

                         if not is_made_straight:
                              # print(f"Debug Straight Check: OESD Found - sequence starting at {unique_ranks[i]}")
                              return "OESD"
                except TypeError:
                     # print(f"Debug: TypeError comparing ranks in OESD check: {unique_ranks}")
                     continue # Skip if ranks are not comparable numbers

            # --- Check for Gutshot (4 ranks out of 5 needed) ---
            # Iterate through all combinations of 4 unique ranks
            from itertools import combinations
            for combo4 in combinations(unique_ranks, 4):
                try:
                     sorted_combo = sorted(list(combo4))
                     diffs = [sorted_combo[j+1] - sorted_combo[j] for j in range(3)]
                     # Gutshot patterns: [1,1,2], [1,2,1], [2,1,1] for the gaps
                     # Or Ace-low/high specific checks
                     is_gutshot = False
                     if diffs == [1, 1, 2]: # e.g., 5, 6, 7, 9 (needs 8)
                         if sorted_combo[2] + 1 not in unique_ranks: is_gutshot = True
                     elif diffs == [1, 2, 1]: # e.g., 5, 6, 8, 9 (needs 7)
                         if sorted_combo[1] + 1 not in unique_ranks: is_gutshot = True
                     elif diffs == [2, 1, 1]: # e.g., 5, 7, 8, 9 (needs 6)
                          if sorted_combo[0] + 1 not in unique_ranks: is_gutshot = True
                     # Special Ace-low cases (e.g., A,2,3,5 needs 4)
                     elif sorted_combo == [1, 2, 3, 5] and 4 not in unique_ranks: is_gutshot = True
                     elif sorted_combo == [1, 2, 4, 5] and 3 not in unique_ranks: is_gutshot = True
                     elif sorted_combo == [1, 3, 4, 5] and 2 not in unique_ranks: is_gutshot = True
                     # Special Ace-high cases (e.g., T,J,Q,A needs K)
                     elif sorted_combo == [10, 11, 12, 14] and 13 not in unique_ranks: is_gutshot = True
                     elif sorted_combo == [10, 11, 13, 14] and 12 not in unique_ranks: is_gutshot = True
                     elif sorted_combo == [10, 12, 13, 14] and 11 not in unique_ranks: is_gutshot = True
                     # Note: T-J-Q-K needing A is OESD (handled above)

                     if is_gutshot:
                          # print(f"Debug Straight Check: Gutshot Found - combo {sorted_combo}")
                          return "GUTSHOT"
                except TypeError:
                     # print(f"Debug: TypeError comparing ranks in Gutshot check: {combo4}")
                     continue # Skip if ranks are not comparable

        except Exception as e_straight:
             print(f"Error during straight draw check logic: {e_straight}")

        return None # No straight draw detected


    # --- Bet Sizing ---
    # REVISED: Calculates the *total* bet amount desired.
    def _calculate_total_bet_amount(self, current_bet_to_call: int, big_blind: int, pot: int, win_rate: float, my_stack: int) -> int:
        """
        Calculates the desired TOTAL bet amount for the current round
        based on pot size, aggression, win rate, and stack constraints.
        """
        # --- Calculate Raise Delta ---
        # Base delta related to pot and aggression
        base_raise_factor = 0.4 + (self.aggression * 0.6) # Factor between 0.4 and 1.0
        raise_amount_delta = pot * base_raise_factor

        # Modify delta based on hand strength
        strength_multiplier = 1.0 + (win_rate - 0.5) * 1.2 # Scale size: Moderate scaling
        raise_amount_delta *= max(0.6, strength_multiplier) # Clamp multiplier effect

        # Apply random jitter
        jitter = 1.0 + (random.random() - 0.5) * 0.2 # +/- 10% jitter
        raise_amount_delta *= jitter

        # --- Minimum and Maximum Raise Delta Constraints ---
        # Minimum delta: Big Blind (simplification, actual rules are complex)
        min_raise_delta = big_blind

        # Maximum delta: Capped by a multiplier of the pot
        max_raise_delta = pot * (1 + self.MAX_RAISE_THRESHOLD_MULT) # e.g., up to 1.5x pot raise delta

        # Clamp the calculated delta
        final_raise_delta = max(min_raise_delta, int(raise_amount_delta))
        final_raise_delta = min(final_raise_delta, int(max_raise_delta))

        # --- Calculate Total Bet Amount ---
        # If betting (no call amount), the bet is the delta (but at least BB)
        if current_bet_to_call == 0:
             desired_total_bet = max(big_blind, final_raise_delta)
        # If raising, the total bet is call amount + delta
        else:
             desired_total_bet = current_bet_to_call + final_raise_delta

        # --- Apply Stack Constraint ---
        # The total bet cannot exceed the player's stack
        final_total_bet_amount = min(desired_total_bet, my_stack)

        # --- Final Legal Minimum Check (redundant with action method, but safe) ---
        if current_bet_to_call > 0: # If raising
            min_legal_total_raise = current_bet_to_call + big_blind
            final_total_bet_amount = max(final_total_bet_amount, min_legal_total_raise)
        else: # If betting
            final_total_bet_amount = max(final_total_bet_amount, big_blind)


        # Ensure final amount doesn't exceed stack AFTER legal min check
        final_total_bet_amount = min(final_total_bet_amount, my_stack)

        # print(f"Debug CalcBet: Pot={pot}, Call={current_bet_to_call}, WR={win_rate:.2f}, Aggro={self.aggression:.2f} -> Delta={final_raise_delta}, DesiredTotal={desired_total_bet}, FinalTotal={final_total_bet_amount}, Stack={my_stack}")
        return int(final_total_bet_amount)


    # --- Opponent Modeling (Keep As Is, ensure PlayerAction used matches stub/import) ---
    def _update_opponent_models(self, action_history: list, num_players: int) -> None:
        """ Updates internal models of opponents based on their actions. """
        # Initialize models if needed
        my_idx = self.my_index # Use cached index
        for i in range(num_players):
            if i != my_idx and i not in self.opponent_models:
                self.opponent_models[i] = {
                    "fold_frequency": 0.0, "aggression_ratio": 0.0, "aggressive": False,
                    "actions": [], "total_actions": 0, "fold_count": 0, "aggressive_count": 0
                 }

        # Process history (assuming tuple format: (player_idx, PlayerAction, amount))
        temp_actions = {i: [] for i in self.opponent_models}
        for item in action_history:
             try: # Add try-except for safer history processing
                 if isinstance(item, (tuple, list)) and len(item) >= 2:
                     p_index, action = item[0], item[1]
                     # Validate action is potentially a PlayerAction enum/value
                     # (Difficult without knowing exact type, basic range check if using ints)
                     is_valid_action = isinstance(action, PlayerAction) if 'PlayerAction' in globals() else (isinstance(action, int) and 0 <= action <= 5)

                     if p_index in self.opponent_models and is_valid_action:
                         temp_actions[p_index].append(action)
                 # else: # Optional debug for skipped items
                     # print(f"Debug: Skipping unexpected action history item format: {item}")
             except Exception as e_hist:
                 print(f"Warning: Error processing action history item '{item}': {e_hist}")
                 continue

        # Update models and recalculate stats
        for i in self.opponent_models:
            model = self.opponent_models[i]
            model["actions"].extend(temp_actions[i])
            model["actions"] = model["actions"][-self.HISTORY_LENGTH:] # Keep history limited

            actions_in_history = model["actions"]
            total_hist_actions = len(actions_in_history)
            model["total_actions"] = total_hist_actions

            if total_hist_actions > 0:
                # Safely get PlayerAction values (use defaults if stub/import failed)
                fold_action = getattr(PlayerAction, 'FOLD', 0)
                bet_action = getattr(PlayerAction, 'BET', 3)
                raise_action = getattr(PlayerAction, 'RAISE', 4)
                all_in_action = getattr(PlayerAction, 'ALL_IN', 5)
                aggressive_actions = {bet_action, raise_action, all_in_action}

                model["fold_count"] = actions_in_history.count(fold_action)
                model["aggressive_count"] = sum(1 for a in actions_in_history if a in aggressive_actions)

                model["fold_frequency"] = model["fold_count"] / total_hist_actions
                model["aggression_ratio"] = model["aggressive_count"] / total_hist_actions
                model["aggressive"] = model["aggression_ratio"] > self.AGGRESSION_THRESHOLD
            else:
                 model.update({
                     "fold_frequency": 0.0, "aggression_ratio": 0.0, "aggressive": False,
                     "fold_count": 0, "aggressive_count": 0
                 })


    # --- Aggression Adjustment (Keep As Is) ---
    def _adjust_aggression(self, opponent_models: dict) -> None:
         """Adjusts the AI's aggression based on opponent tendencies."""
         num_opponents = len(opponent_models)
         if num_opponents == 0: return

         try: # Add try-except for safety
             aggressive_opponents = sum(1 for model in opponent_models.values() if model.get("aggressive", False))
             passive_opponents = num_opponents - aggressive_opponents

             aggression_change = 0.0 # Use float
             # Check division by zero
             if num_opponents > 0:
                 if passive_opponents / num_opponents > 0.6:
                     aggression_change = self.AGGRESSION_ADJUSTMENT_RATE
                 elif aggressive_opponents / num_opponents > 0.5:
                     aggression_change = -self.AGGRESSION_ADJUSTMENT_RATE / 1.5

                 # Simplified avg aggression check
                 avg_opp_aggression = sum(m.get('aggression_ratio', 0.0) for m in opponent_models.values()) / num_opponents
                 if avg_opp_aggression > self.AGGRESSION_THRESHOLD + 0.1:
                      aggression_change += self.RAISEPLAYER_DETECTED_AGGRESSION_BOOST * 0.1

             self.aggression += aggression_change
             self.aggression = max(self.MIN_RAISE_AGGRESSION, min(self.aggression, 0.95))
             # print(f"Debug Aggression Adjust: Opponents={num_opponents}, Aggro={aggressive_opponents}, Passive={passive_opponents}, Change={aggression_change:.3f}, NewAggro={self.aggression:.3f}")

         except Exception as e_aggro:
             print(f"Warning: Error adjusting aggression: {e_aggro}")


    # --- Game/Tournament Tracking (Keep As Is) ---
    def update_winning_percentage(self, won: bool):
        """Updates the AI's tournament winning percentage."""
        self.total_games += 1
        self.wins += 1 if won else 0
        if self.total_games > 0:
             self.winning_percentage = (self.wins / self.total_games) * 100
        else:
             self.winning_percentage = 0.0

    def reset_game_stats(self):
        """Resets the game statistics, e.g., for a new tournament."""
        self.total_games = 0
        self.wins = 0
        self.winning_percentage = 0.0
        self.opponent_models = {}
        self.aggression = self.STARTING_AGGRESSION


# --- Example Usage (Keep As Is, ensure stubs are sufficient) ---
if __name__ == '__main__':
    # ... (Keep the existing test cases) ...
    # Add a BET scenario test
    print("\n--- Test Case 5: BET Scenario (First to Act) ---")
    try:
        player_bet = AdaptiveRaisePlayer("BetBot", 1000)
        player_bet.my_index = 0
        hole_bet = [Card(Rank.ACE, Suit.CLUBS), Card(Rank.KING, Suit.CLUBS)] # Strong hand
        community_bet = [Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.TEN, Suit.SPADES), Card(Rank.TWO, Suit.HEARTS)]
        game_state_bet = [
            hole_bet[0].get_index(), hole_bet[1].get_index(),
            community_bet[0].get_index(), community_bet[1].get_index(), community_bet[2].get_index(), -1, -1,
            30, # Pot (Blinds only)
            0,  # Current Bet to Call is 0
            10, # Big blind
            0,  # My index
            3,  # Num players
            990, 1000, 1010 # Stacks (Assume SB posted 5, BB posted 10)
        ]
        player_bet.set_hole_cards(hole_bet) # Use dummy method if needed
        action_tuple = player_bet.action(game_state_bet, [])
        print(f"Action (Bet Scenario): {action_tuple}") # Expecting (PlayerAction.BET, amount > BB)

        print("\n--- Test Case 6: RAISE Scenario ---")
        # Same hand, but now facing a bet
        action_history_raise = [(1, PlayerAction.BET, 30)] # Player 1 bets 30
        game_state_raise = [
            hole_bet[0].get_index(), hole_bet[1].get_index(),
            community_bet[0].get_index(), community_bet[1].get_index(), community_bet[2].get_index(), -1, -1,
            30 + 30, # Pot (Blinds + Bet)
            30,  # Current Bet to Call is 30
            10, # Big blind
            0,  # My index
            3,  # Num players
            990, 970, 1010 # Stacks (P0, P1(bet 30), P2)
        ]
        action_tuple_raise = player_bet.action(game_state_raise, action_history_raise)
        print(f"Action (Raise Scenario): {action_tuple_raise}") # Expecting (PlayerAction.RAISE, amount > 30+BB)

        print("\n--- Test Case 7: All-In (Forced Call) ---")
        action_history_allin = [(1, PlayerAction.BET, 1000)] # Player 1 bets large
        game_state_allin = [
            hole_bet[0].get_index(), hole_bet[1].get_index(),
            community_bet[0].get_index(), community_bet[1].get_index(), community_bet[2].get_index(), -1, -1,
            30 + 1000, # Pot
            1000,  # Current Bet to Call is 1000
            10, # Big blind
            0,  # My index
            3,  # Num players
            990, 0, 1010 # Stacks (P0, P1(all-in), P2) -> P0 has 990 stack
        ]
        action_tuple_allin = player_bet.action(game_state_allin, action_history_allin)
        print(f"Action (Forced All-In Call): {action_tuple_allin}") # Expecting (PlayerAction.ALL_IN, 990)

    except NameError as e:
        print(f"Could not run tests due to missing definitions (likely Card/Rank/Suit): {e}")
    except AttributeError as e:
         print(f"Could not run tests due to missing attributes (likely Card/Rank/Suit structure error): {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")