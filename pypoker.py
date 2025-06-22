import random
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.deck import Deck
from pypokerengine.engine.card import Card

# --- Monte Carlo Configuration ---
# !! TUNE THIS BASED ON YOUR SYSTEM SPEED AND TIME LIMITS !!
# Lower values are faster but less accurate. Higher values are slower.
# Start around 100-300 and adjust.
NB_SIMULATION = 200

# --- Strategy Thresholds & Parameters ---
# Post-flop Win Rate Thresholds (from Monte Carlo)
POSTFLOP_RAISE_THRESHOLD_STRONG = 0.80 # Win rate above which we raise strongly/consider all-in
POSTFLOP_RAISE_THRESHOLD_GOOD = 0.60  # Win rate above which we are happy to raise/bet value
POSTFLOP_CALL_THRESHOLD = 0.45       # Win rate above which we might call (if odds are okay)
POSTFLOP_BLUFF_CHECKED_TO_CHANCE = 0.15 # Chance to bluff bet when action is checked to us

# Pre-flop Hand Strength Categories (Heuristic)
# Using simple classification here as MC is too slow pre-flop
PREFLOP_RAISE_CATEGORIES = ["premium", "strong"]
PREFLOP_CALL_CATEGORIES = ["good"]
# Speculative/Weak hands will generally fold to raises, might limp/call small bets in position

# Sizing Parameters
PREFLOP_OPEN_RAISE_BB = 3   # Standard opening raise size in Big Blinds
POSTFLOP_BET_POT_RATIO_VALUE = 0.65 # Bet/Raise ~65% of pot for value
POSTFLOP_BET_POT_RATIO_BLUFF = 0.45 # Bet/Raise ~45% of pot for bluffs
POSTFLOP_RAISE_MULTIPLIER = 2.5 # Raise to ~2.5x the previous bet/raise total

# Stack Management
PREFLOP_ALLIN_THRESHOLD_BB = 15 # Below this many BBs, play push/fold pre-flop

class MonteCarloRaisePlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        self.hole_card_obj = []
        self.my_stack = 0
        self.my_uuid = None
        self.big_blind = 10 # Default, will be updated

    # --- Required pypokerengine methods ---

    def declare_action(self, valid_actions, hole_card, round_state):
        # -- 1. Parse Game State --
        community_cards_str = round_state['community_card']
        self.hole_card_obj = gen_cards(hole_card) # Convert str to Card objects
        community_card_obj = gen_cards(community_cards_str)
        pot_size = round_state['pot']['main']['amount']
        my_player_info = round_state['seats'][self.seat_id]
        self.my_stack = my_player_info['stack']
        # Calculate current bet to call (amount needed TO MATCH highest bet)
        street = round_state['street']
        current_total_bet = 0
        my_round_contribution = 0

        # Find the highest bet amount currently on the table for this street
        # Also find how much *we* have contributed this street
        if street != 'preflop': # Easier post-flop
            action_histories = round_state.get('action_histories', {}).get(street, [])
            player_bets_this_street = {}
            highest_bet_this_street = 0
            for action in action_histories:
                uuid = action['uuid']
                amount = action.get('amount', 0)
                action_type = action['action']
                if action_type in ['BET', 'RAISE']:
                    player_bets_this_street[uuid] = amount # Total bet amount for player
                    highest_bet_this_street = max(highest_bet_this_street, amount)
                elif action_type == 'CALL':
                    # If someone calls, their total bet matches the highest bet so far
                    player_bets_this_street[uuid] = highest_bet_this_street

            current_total_bet = highest_bet_this_street
            my_round_contribution = player_bets_this_street.get(self.uuid, 0)

        else: # Preflop needs careful handling of blinds
            # Find highest total commitment preflop
             max_commitment = 0
             for seat in round_state['seats']:
                  max_commitment = max(max_commitment, seat['round_bet'])
             current_total_bet = max_commitment
             my_round_contribution = my_player_info['round_bet']


        call_amount = current_total_bet - my_round_contribution
        can_call = call_amount <= self.my_stack and any(a['action'] == 'call' for a in valid_actions)
        can_raise = any(a['action'] == 'raise' for a in valid_actions)
        can_check = call_amount == 0 and not any(a['action'] == 'call' for a in valid_actions) # Check only if call=0
        can_bet = call_amount == 0 and any(a['action'] == 'bet' for a in valid_actions) # Can bet if check is possible


        num_active_players = len([p for p in round_state['seats'] if p['state'] == 'participating' and p['uuid'] != self.uuid])

        # Min/Max raise amounts from valid_actions
        min_raise, max_raise = 0, self.my_stack + my_round_contribution # Default max is all-in
        if can_raise:
            raise_action_info = next(a for a in valid_actions if a['action'] == 'raise')
            min_raise = raise_action_info['amount']['min']
            max_raise = raise_action_info['amount']['max']
        elif can_bet: # Similar logic if betting is possible
             bet_action_info = next(a for a in valid_actions if a['action'] == 'bet')
             # PyPokerEngine uses 'amount' directly for bet min/max usually
             # Assuming min bet is BB and max is stack for simplicity if not specified
             min_raise = bet_action_info.get('amount', {}).get('min', self.big_blind)
             max_raise = bet_action_info.get('amount', {}).get('max', self.my_stack)


        # -- 2. Decide Action based on Phase --
        action = 'fold' # Default safety action
        amount = 0

        if street == 'preflop':
            action, amount = self._decide_preflop(
                call_amount, can_check, can_call, can_raise,
                min_raise, max_raise, pot_size
            )
        else: # Flop, Turn, River
            action, amount = self._decide_postflop(
                community_card_obj, num_active_players,
                call_amount, can_check, can_call, can_raise, can_bet,
                min_raise, max_raise, pot_size, current_total_bet
            )

        # -- 3. Validate and Return Action --
        # Ensure the chosen action is valid and amount is within bounds
        final_action = 'fold'
        final_amount = 0

        if action == 'raise' and can_raise:
            final_action = 'raise'
            # Amount should be the TOTAL bet amount after raising
            final_amount = max(min_raise, min(amount, max_raise))
            # Ensure we don't raise less than call amount if call exists
            if can_call and final_amount < call_amount:
                final_action = 'call'
                final_amount = call_amount

        elif action == 'bet' and can_bet:
            final_action = 'bet'
            # Amount is the bet amount itself
            final_amount = max(min_raise, min(amount, max_raise)) # Use min/max from 'bet' if available

        elif action == 'call' and can_call:
            final_action = 'call'
            final_amount = call_amount

        elif action == 'check' and can_check:
             final_action = 'check'
             final_amount = 0

        # If the desired action isn't valid, fall back
        if final_action == 'fold': # Default or if raise/call impossible
             if can_check:
                 final_action = 'check'
                 final_amount = 0
             elif can_call and call_amount <= self.my_stack:
                 # Basic fallback: Call if we wanted to raise but couldn't, and call is possible/cheap?
                 # This logic could be much more sophisticated
                 if action == 'raise' and call_amount < self.my_stack * 0.2: # Only call if cheap
                     final_action = 'call'
                     final_amount = call_amount
                 else:
                     final_action = 'fold'
                     final_amount = 0
             else:
                 final_action = 'fold'
                 final_amount = 0


        # Final sanity check on amount
        if final_action != 'fold' and final_action != 'check':
            final_amount = max(0, final_amount) # Cannot be negative
            # Ensure final amount doesn't exceed stack (when considering amount already bet)
            amount_to_add = final_amount - my_round_contribution
            if amount_to_add > self.my_stack:
                 # This implies going all-in
                 if final_action == 'raise':
                      final_amount = self.my_stack + my_round_contribution
                 elif final_action == 'call':
                      final_amount = min(call_amount, self.my_stack + my_round_contribution) # Call capped at stack
                 elif final_action == 'bet':
                      final_amount = self.my_stack # Bet capped at stack

        return final_action, final_amount

    def game_start(self, game_info):
        self.my_uuid = game_info['seats'][self.seat_id]['uuid']
        self.big_blind = game_info['rule']['ante'] * 2 # Approximation if BB not directly available
        # print(f"DEBUG: Game start. My UUID: {self.my_uuid}, Seat: {self.seat_id}")
        pass

    def round_start(self, round_count, hole_card, seats):
        self.hole_card_obj = [] # Reset hole cards
        my_seat_info = next(s for s in seats if s['uuid'] == self.uuid)
        self.seat_id = seats.index(my_seat_info) # Store seat index
        # print(f"DEBUG: Round start. My Seat ID: {self.seat_id}")
        pass

    def round_result(self, players, round_count, hand_info):
        pass

    def game_update(self, new_state, action):
        # Could use this to track opponent actions for modeling later
        pass

    def game_result(self, game_info):
        pass

    # --- Helper Methods ---

    def _get_game_phase(self, community_cards: List[Card]) -> str:
        """Determines the current game phase string."""
        num_community = len(community_cards)
        if num_community == 0: return 'preflop'
        elif num_community == 3: return 'flop'
        elif num_community == 4: return 'turn'
        elif num_community == 5: return 'river'
        else: return 'unknown'

    def _classify_preflop_hand(self) -> str:
        """ Classifies starting hand strength heuristically. """
        if not self.hole_card_obj or len(self.hole_card_obj) != 2: return "weak"
        # Use pypokerengine's Card objects directly
        c1, c2 = sorted(self.hole_card_obj, key=lambda c: c.rank, reverse=True)
        r1, r2 = c1.rank, c2.rank # Ranks are 2-14 (A=14)
        s1, s2 = c1.suit, c2.suit
        suited = (s1 == s2)
        is_pair = (r1 == r2)

        # Premium: AA, KK, QQ, JJ, AKs, AKo
        if is_pair and r1 >= 11: return "premium" # JJ+
        if r1 == 14 and r2 == 13: return "premium" # AK

        # Strong: TT, AQs, AQo, KQs, AJs, KJs
        if is_pair and r1 == 10: return "strong" # TT
        if r1 == 14 and r2 == 12: return "strong" # AQ
        if r1 == 13 and r2 == 12 and suited: return "strong" # KQs
        if r1 == 14 and r2 == 11: return "strong" # AJ
        if r1 == 13 and r2 == 11 and suited: return "strong" # KJs

        # Good: 99-77, Axs (suited Aces), KTs+, QTs+, JTs, T9s, 98s, 87s, 76s, ATo+, KQo
        if is_pair and r1 >= 7: return "good" # 77-99
        if r1 == 14 and suited: return "good" # Axs
        if r1 == 13 and r2 >= 10 and suited: return "good" # KTs+ suited
        if r1 == 12 and r2 >= 10 and suited: return "good" # QTs+ suited
        if r1 == 11 and r2 == 10 and suited: return "good" # JTs suited
        if r1 == 10 and r2 == 9 and suited: return "good" # T9s
        if r1 == 9 and r2 == 8 and suited: return "good" # 98s
        if r1 == 8 and r2 == 7 and suited: return "good" # 87s
        if r1 == 7 and r2 == 6 and suited: return "good" # 76s
        if r1 == 14 and r2 >= 10 and not suited: return "good" # ATo+ offsuit
        if r1 == 13 and r2 == 12 and not suited: return "good" # KQo

        # Speculative: Mid/low pairs, suited connectors/gappers, Ax offsuit
        if is_pair: return "speculative" # 22-66
        if suited and (r1 - r2 <= 4): return "speculative" # Suited connectors/gappers
        if r1 == 14: return "speculative" # A2o-A9o

        return "weak" # Everything else

    def _run_monte_carlo(self, community_card_obj, num_opponents, num_simulations) -> float:
        """Runs Monte Carlo simulation to estimate win rate."""
        wins = 0
        ties = 0

        # Check for impossible scenarios
        if num_opponents <= 0:
            return 1.0 # Assume win if no opponents left

        for _ in range(num_simulations):
            # Create a deck copy and remove known cards
            deck = Deck()
            known_cards = self.hole_card_obj + community_card_obj
            deck.shuffle()
            # Need to remove cards by string representation if Deck uses strings
            known_card_strs = [str(c) for c in known_cards]
            deck.deck = [c for c in deck.deck if str(c) not in known_card_strs]


            # Deal opponent hands and remaining community cards
            opponent_hands = []
            all_sim_community = list(community_card_obj) # Start with current community cards

            try:
                # Deal opponent hands
                for _ in range(num_opponents):
                    if len(deck.deck) < 2: raise ValueError("Not enough cards for opponent hands")
                    hand = [deck.draw_card(), deck.draw_card()]
                    opponent_hands.append(hand)

                # Deal remaining community cards
                needed_community = 5 - len(all_sim_community)
                if len(deck.deck) < needed_community: raise ValueError("Not enough cards for community")
                for _ in range(needed_community):
                    all_sim_community.append(deck.draw_card())

                # Evaluate hands
                my_best_hand_info = HandEvaluator.gen_hand_rank_info(self.hole_card_obj, all_sim_community)
                my_rank = my_best_hand_info['rank']
                #print(f"DEBUG MC Eval: My Hand: {my_best_hand_info['hand']['high']}, Rank: {my_rank}")

                opponent_ranks = []
                for opp_hand in opponent_hands:
                     opp_best_hand_info = HandEvaluator.gen_hand_rank_info(opp_hand, all_sim_community)
                     opponent_ranks.append(opp_best_hand_info['rank'])
                     #print(f"DEBUG MC Eval: Opp Hand: {opp_best_hand_info['hand']['high']}, Rank: {opp_best_hand_info['rank']}")


                # Determine win/tie/loss
                won_sim = True
                tied_sim = False
                best_opp_rank = 0
                if opponent_ranks:
                    best_opp_rank = max(opponent_ranks)

                if my_rank < best_opp_rank:
                    won_sim = False
                elif my_rank == best_opp_rank:
                    # Need detailed tie-breaking (HandEvaluator might handle this implicitly if ranks are detailed enough)
                    # Simplified: If ranks are equal, check kickers etc. Let's assume HandEvaluator's rank IS detailed.
                    # Re-evaluate against *only* the best opponent hands if ranks are equal
                    opponents_with_best_rank = [
                        opp_hand for i, opp_hand in enumerate(opponent_hands) if opponent_ranks[i] == my_rank
                    ]
                    is_strictly_better = True
                    for opp_hand in opponents_with_best_rank:
                        # gen_hand_rank_info returns detailed rank. Higher is better.
                        opp_eval = HandEvaluator.gen_hand_rank_info(opp_hand, all_sim_community)
                        # Need to compare detailed hand strength (e.g. using HandEvaluator.judge_hand method if available or comparing ranks)
                        # Assuming simple rank comparison: if any opponent rank is equal, it's potentially a tie or loss on kickers
                        # For simplicity here, if ranks are equal, we call it a tie. A full eval needs kicker comparison.
                        if opp_eval['rank'] == my_rank: # Simple tie condition
                             is_strictly_better = False
                             break # Found at least one opponent matching our rank

                    if not is_strictly_better:
                        won_sim = False
                        tied_sim = True # We didn't lose, but didn't strictly win
                    # If is_strictly_better remains True, we won (e.g. our pair had better kicker)

                if won_sim:
                    wins += 1
                elif tied_sim:
                    ties += 1

            except ValueError as e:
                # print(f"WARN: Monte Carlo simulation error (not enough cards?): {e}")
                # Skip this simulation iteration if deck runs out
                continue
            except Exception as e:
                # print(f"WARN: Unexpected error in Monte Carlo simulation: {e}")
                continue # Skip simulation on other errors


        if num_simulations == 0: return 0.0
        # Calculate win rate (treat ties as half a win)
        win_rate = (wins + (ties / 2.0)) / num_simulations
        #print(f"DEBUG: MC Result - Wins: {wins}, Ties: {ties}, Rate: {win_rate:.3f}")
        return win_rate

    def _calculate_total_bet_amount(self, pot_size: int, is_value: bool, current_total_bet: int = 0) -> int:
        """Calculates the TOTAL bet/raise amount based on pot."""
        ratio = POSTFLOP_BET_POT_RATIO_VALUE if is_value else POSTFLOP_BET_POT_RATIO_BLUFF
        # For raises, consider the pot size *after* the call
        effective_pot = pot_size + current_total_bet # Approximation of pot if bet is called
        target_bet = int(effective_pot * ratio)

        # Ensure minimum bet (at least BB)
        target_bet = max(self.big_blind, target_bet)
        return target_bet

    def _calculate_total_raise_amount(self, pot_size: int, current_total_bet: int, min_raise_total: int) -> int:
         """Calculates the TOTAL amount to raise to."""
         # Option 1: Raise based on pot size
         pot_based_raise = self._calculate_total_bet_amount(pot_size, True, current_total_bet)

         # Option 2: Raise based on multiplier of current bet
         # Amount to call + raise increment
         # raise_increment = (current_total_bet * (POSTFLOP_RAISE_MULTIPLIER -1))
         # multiplier_based_raise_total = current_total_bet + raise_increment # Incorrect logic
         multiplier_based_raise_total = int(current_total_bet * POSTFLOP_RAISE_MULTIPLIER)


         # Choose a reasonable raise size (e.g., average or max)
         target_total = max(pot_based_raise, multiplier_based_raise_total)

         # Ensure it meets the minimum required raise
         target_total = max(min_raise_total, target_total)

         return int(target_total)


    def _decide_preflop(self, call_amount, can_check, can_call, can_raise, min_raise, max_raise, pot_size):
        """Pre-flop decision logic using heuristics."""
        strength_category = self._classify_preflop_hand()
        my_stack_bb = self.my_stack / self.big_blind if self.big_blind > 0 else 0

        # Push/Fold for short stacks
        if my_stack_bb <= PREFLOP_ALLIN_THRESHOLD_BB:
            if strength_category in ["premium", "strong"] or (strength_category == "good" and my_stack_bb <= 10):
                # Go all-in
                 if can_raise: return 'raise', max_raise # Max raise is all-in
                 elif can_call: return 'call', call_amount # Call if raise not allowed (already all-in?)
                 else: return 'fold', 0 # Should not happen if stack > 0
            else:
                # Fold weaker hands when short unless checking is free
                return ('check', 0) if can_check else ('fold', 0)

        # Standard Pre-flop
        if can_check or can_bet: # Our action to open (or check BB)
            if strength_category in PREFLOP_RAISE_CATEGORIES:
                 # Raise standard amount
                 target_bet = self.big_blind * PREFLOP_OPEN_RAISE_BB
                 # Adjust if limpers exist (pot is larger than just blinds)
                 if pot_size > self.big_blind * 1.5:
                      target_bet = int(pot_size * 1.5 + self.big_blind) # Pot raise + BB
                 amount = max(min_raise, min(target_bet, max_raise))
                 return 'raise', amount # Use raise even if opening, pypokerengine handles it
            elif strength_category in PREFLOP_CALL_CATEGORIES:
                 # Limp/Call BB - passive, but simple
                 return ('check', 0) if can_check else ('call', call_amount)
            else: # Weak or Speculative
                 return ('check', 0) if can_check else ('fold', 0)

        elif can_call or can_raise: # Facing a bet/raise
            if strength_category in PREFLOP_RAISE_CATEGORIES:
                 # Re-raise (3-bet+)
                 target_raise = self._calculate_total_raise_amount(pot_size, call_amount + self.my_round_contribution, min_raise) # current_total_bet = call_amount + my_round_contribution
                 amount = max(min_raise, min(target_raise, max_raise))
                 return 'raise', amount
            elif strength_category in PREFLOP_CALL_CATEGORIES:
                 # Call if raise isn't too large
                 if call_amount <= self.my_stack * 0.15: # Call if <= 15% of stack
                      return 'call', call_amount
                 else:
                      return 'fold', 0
            else: # Weak or Speculative
                 return 'fold', 0

        return 'fold', 0 # Default safety

    def _decide_postflop(self, community_card_obj, num_opponents, call_amount, can_check, can_call, can_raise, can_bet, min_raise, max_raise, pot_size, current_total_bet):
         """Post-flop decision logic using Monte Carlo."""

         # Estimate win rate using Monte Carlo
         win_rate = self._run_monte_carlo(community_card_obj, num_opponents, NB_SIMULATION)

         # Calculate required equity (pot odds) if facing a bet
         required_equity = 0.0
         if call_amount > 0:
              total_pot_if_called = pot_size + call_amount # Pot includes our call
              if total_pot_if_called > 0:
                   required_equity = call_amount / total_pot_if_called

         action = 'fold'
         amount = 0

         if can_check or can_bet: # Action checked to us
             if win_rate >= POSTFLOP_RAISE_THRESHOLD_GOOD:
                  # Value Bet
                  target_bet = self._calculate_total_bet_amount(pot_size, True)
                  amount = max(min_raise, min(target_bet, max_raise)) # Use min/max raise as proxy for bet bounds
                  action = 'bet'
             elif random.random() < POSTFLOP_BLUFF_CHECKED_TO_CHANCE:
                  # Bluff Bet
                  target_bet = self._calculate_total_bet_amount(pot_size, False)
                  amount = max(min_raise, min(target_bet, max_raise))
                  action = 'bet'
             else:
                  # Check otherwise
                  action = 'check'

         elif can_call or can_raise: # Facing a bet
             if win_rate >= POSTFLOP_RAISE_THRESHOLD_STRONG:
                  # Strong Value Raise / All-in
                  target_raise = self._calculate_total_raise_amount(pot_size, current_total_bet, min_raise)
                  amount = max(min_raise, min(target_raise, max_raise))
                  # Consider all-in if raise is large part of stack
                  if amount >= self.my_stack * 0.7 + self.my_round_contribution: # Committing ~70%+ effective stack
                       amount = max_raise # Go All-in
                  action = 'raise'

             elif win_rate >= POSTFLOP_RAISE_THRESHOLD_GOOD:
                   # Good Value Raise (meets odds comfortably)
                   target_raise = self._calculate_total_raise_amount(pot_size, current_total_bet, min_raise)
                   amount = max(min_raise, min(target_raise, max_raise))
                   action = 'raise'

             elif win_rate >= POSTFLOP_CALL_THRESHOLD and win_rate >= required_equity:
                   # Call threshold met AND getting pot odds
                   action = 'call'
                   amount = call_amount

             elif win_rate >= required_equity * 0.8: # Close odds (Implied odds / Bluff catch region - simplified)
                  # Maybe call small bets, otherwise fold
                  if call_amount < self.my_stack * 0.1: # Only call cheap bets here
                      action = 'call'
                      amount = call_amount
                  else:
                      action = 'fold'
             else:
                  # Fold if not meeting thresholds or odds
                  action = 'fold'

         # Final check: if we decided to raise/bet/call but cannot, fold.
         if action == 'raise' and not can_raise: action = 'call' if can_call else 'fold'
         if action == 'bet' and not can_bet: action = 'check' if can_check else 'fold' # Should not happen if logic is right
         if action == 'call' and not can_call: action = 'fold'
         if action == 'check' and not can_check: action = 'fold' # If check not valid, must fold


         # Ensure amount corresponds to action
         if action == 'call': amount = call_amount
         elif action == 'check' or action == 'fold': amount = 0
         # Amount for raise/bet is already calculated, capped by declare_action

         return action, amount