import random
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict
import math
import time
import copy

# --- Minimal Placeholder Classes (Assuming external definitions exist) ---
# (If these are truly external, you'd import them. Defined here for standalone running.)

class Rank: # Example structure
    ACE = type('RankValue', (), {'value': 14})
    # ... (other ranks needed for Card init) ...
    TWO = type('RankValue', (), {'value': 2})

class Suit: # Example structure
    SPADES = type('SuitValue', (), {})
    HEARTS = type('SuitValue', (), {})
    DIAMONDS = type('SuitValue', (), {})
    CLUBS = type('SuitValue', (), {})

class Card:
    _card_map: Dict[Tuple[int, Suit], 'Card'] = {}
    _index_map: Dict[int, 'Card'] = {}
    _next_index = 0
    _all_ranks = [type('RankValue', (), {'value': v}) for v in range(2, 15)] # 2 to Ace
    _all_suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]

    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
        self._index = Card._index_map.get((rank.value, suit), -1) # Check if exists
        if self._index == -1:
             self._index = Card._next_index
             Card._card_map[(rank.value, suit)] = self
             Card._index_map[self._index] = self
             Card._next_index += 1

    def get_index(self) -> int: return self._index
    def __repr__(self): # Minimal repr
        rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
        r = rank_map.get(self.rank.value, str(self.rank.value))
        s = {Suit.SPADES: 's', Suit.HEARTS: 'h', Suit.DIAMONDS: 'd', Suit.CLUBS: 'c'}.get(self.suit, '?')
        return f"{r}{s}"
    def __lt__(self, other): return self.rank.value < other.rank.value
    def __eq__(self, other): return isinstance(other, Card) and self.rank.value == other.rank.value and self.suit == other.suit
    def __hash__(self): return hash((self.rank.value, self.suit))

# Initialize all cards for indexing
for rank_obj in Card._all_ranks:
    for suit_obj in Card._all_suits:
        Card(rank_obj, suit_obj)

class Deck: # Minimal Deck for indexing/simulation
    def __init__(self): self.cards = list(Card._index_map.values())
    def shuffle(self): random.shuffle(self.cards)

class HandRank: HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH, ROYAL_FLUSH = range(10)
class HandResult: # Minimal HandResult
    def __init__(self, rank: int, value: int): self.rank, self.value = rank, value
    def __gt__(self, other): return (self.rank, self.value) > (other.rank, other.value) if isinstance(other, HandResult) else NotImplemented
    def __ge__(self, other): return (self.rank, self.value) >= (other.rank, other.value) if isinstance(other, HandResult) else NotImplemented
    def __eq__(self, other): return (self.rank, self.value) == (other.rank, other.value) if isinstance(other, HandResult) else NotImplemented

class HandEvaluator: # Minimal Placeholder
    @staticmethod
    def evaluate_hand(hole: List[Card], comm: List[Card]) -> HandResult:
        cards = hole + comm
        if not cards: return HandResult(HandRank.HIGH_CARD, 0)
        ranks = sorted([c.rank.value for c in cards], reverse=True)
        # Extremely simplified evaluation for placeholder needs
        counts = Counter(ranks)
        if any(c == 2 for c in counts.values()): return HandResult(HandRank.PAIR, max(r for r, c in counts.items() if c==2))
        return HandResult(HandRank.HIGH_CARD, ranks[0] if ranks else 0)

class PlayerAction: FOLD, CHECK, CALL, BET, RAISE, ALL_IN = range(6)
class PlayerStatus: OUT, FOLDED, ACTIVE, ALL_IN = range(4)
class GamePhase: SETUP, PRE_FLOP, FLOP, TURN, RIVER, SHOWDOWN = range(6)

class Player: # Base Player
     def __init__(self, name: str, stack: int):
         self.name = name
         self.stack = stack
         self.hole_cards: List[Card] = []
         self.bet_amount = 0 # Bet in current round
         self.status = PlayerStatus.ACTIVE
     def action(self, gs: list, hist: list) -> Tuple[int, int]: raise NotImplementedError
     def reset_for_new_hand(self):
        self.hole_cards = []
        self.bet_amount = 0
        if self.stack > 0: self.status = PlayerStatus.ACTIVE
        else: self.status = PlayerStatus.OUT
     def reset_game_stats(self): pass
     def update_winning_percentage(self, won: bool): pass

# --- Helper: Card Indexing ---
def card_from_index(index: int) -> Optional[Card]: return Card._index_map.get(index)

# --- MCTS Node ---
class MCTSNode:
    """Represents an MCTS node."""
    __slots__ = ['game_state', 'action_history', 'parent', 'children', 'n', 'q',
                 'action', 'amount', 'ai_player_index', 'player_bets_in_round',
                 'current_player_index', '_terminal'] # Memory optimization

    def __init__(self, gs: list, hist: list, parent: Optional['MCTSNode'], ai_idx: int, bets: Dict[int, int], cur_player: int, act: Optional[int] = None, amt: int = 0):
        self.game_state: list = gs
        self.action_history: list = hist
        self.parent: Optional['MCTSNode'] = parent
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.n: int = 0
        self.q: float = 0.0
        self.action: Optional[int] = act
        self.amount: int = amt
        self.ai_player_index: int = ai_idx
        self.player_bets_in_round: Dict[int, int] = bets
        self.current_player_index: int = cur_player
        self._terminal: Optional[bool] = None

    @property
    def is_terminal(self) -> bool:
        if self._terminal is None: self._check_terminal()
        return self._terminal

    def _check_terminal(self):
        """Simplified terminal check for MCTS."""
        num_p = self.game_state[11]
        stacks = self.game_state[12:12+num_p]
        # Quick check on active players (stack > 0)
        active_count = sum(1 for s in stacks if s > 0)
        community_count = sum(1 for i in self.game_state[2:7] if i is not None and i >= 0)
        self._terminal = (active_count <= 1 or community_count == 5)

    def is_fully_expanded(self, num_p: int, ai_player: 'MonteCarloGTOAIPlayer') -> bool:
        if self.is_terminal: return True
        node_player_idx = self.current_player_index
        stacks = self.game_state[12:12+num_p]
        if not (0 <= node_player_idx < num_p) or stacks[node_player_idx] <= 0: return True

        stack = stacks[node_player_idx]
        bet = self.player_bets_in_round.get(node_player_idx, 0)
        possible_actions = ai_player._get_possible_actions(
            self.game_state[8], self.game_state[7], stack, self.game_state[9], bet # Ctx: current_bet, pot, stack, bb, player_bet
        )
        possible_keys = {ai_player._standardize_action_tuple(a, amt, stack) for a, amt in possible_actions}
        return len(self.children) >= len(possible_keys)

# --- Main AI Player Class ---
class MonteCarloGTOAIPlayer(Player):
    """Optimized MCTS AI Player."""
    # --- Config ---
    MCTS_SIMULATIONS = 80 # Further reduced
    EXPLORATION_CONSTANT = 1.414
    MAX_MCTS_DEPTH = 4     # Further reduced
    HISTORY_LENGTH = 8

    def __init__(self, name: str, stack: int):
        super().__init__(name, stack)
        self.winning_percentage = 0.0
        self.total_games = 0
        self.wins = 0
        # Simplified history tracking - store only relevant info if needed
        # self.opponent_history = defaultdict(list)
        # self.opponent_fold_frequency = defaultdict(float)
        self.my_index = -1 # Initialize safely
        self._current_bb = 0

    def reset_for_new_hand(self):
        super().reset_for_new_hand()
        # No AI-specific state needs reset per hand currently

    def action(self, game_state: list[int], action_history: list) -> Tuple[int, int]:
        """Determines AI action using MCTS."""
        start_time = time.time()

        try: # Robust state parsing
            self.my_index = game_state[10]
            num_players = game_state[11]
            player_stacks = game_state[12:12 + num_players]
            my_stack = player_stacks[self.my_index]
            self.hole_cards = [card_from_index(i) for i in game_state[0:2] if card_from_index(i)] # More direct
            current_bet = game_state[8]
            pot = game_state[7]
            self._current_bb = game_state[9]
        except (IndexError, TypeError):
            print(f"Error: {self.name} invalid game_state. Folding.")
            return PlayerAction.FOLD, 0

        if len(self.hole_cards) != 2:
            print(f"AI {self.name} invalid hole cards. Folding/Checking.")
            call = max(0, current_bet - self.bet_amount)
            return (PlayerAction.FOLD, 0) if call > 0 else (PlayerAction.CHECK, 0)

        # --- MCTS ---
        best_action, best_amount = self._monte_carlo_tree_search(game_state, num_players)

        # --- Validation ---
        validated_action, validated_amount = self._validate_action(
            best_action, best_amount, my_stack, current_bet, self.bet_amount
        )

        # print(f"MCTS {self.name}: {validated_action} {validated_amount} ({time.time() - start_time:.3f}s)") # Debug
        return validated_action, validated_amount

    def _validate_action(self, action: int, amount: int, stack: int, current_bet: int, my_bet_in_round: int) -> Tuple[int, int]:
        """Ensures the chosen action is legal."""
        call_amount = max(0, current_bet - my_bet_in_round)
        can_check = (call_amount == 0)
        min_bb = self._current_bb if self._current_bb > 0 else 1

        # Handle invalid stack or non-positive amount for betting actions
        if stack <= 0: return PlayerAction.CHECK, 0 # Should already be OUT or ALL_IN
        if action not in [PlayerAction.FOLD, PlayerAction.CHECK] and amount < 0: amount = 0

        cost = 0
        final_action = action
        final_amount = amount

        if action == PlayerAction.FOLD: final_amount = 0
        elif action == PlayerAction.CHECK:
            final_amount = 0
            if not can_check: final_action = PlayerAction.FOLD # Force fold if check illegal
        elif action == PlayerAction.CALL:
            if call_amount == 0: final_action, final_amount = PlayerAction.CHECK, 0 # Treat as check
            else: cost = min(call_amount, stack); final_amount = cost
        elif action == PlayerAction.BET:
            if not can_check: final_action = PlayerAction.FOLD # Force fold if bet illegal
            else:
                bet_value = max(min_bb, amount)
                cost = min(bet_value, stack); final_amount = cost
        elif action == PlayerAction.RAISE:
            if call_amount <= 0: # Treat as BET if possible, else FOLD
                if can_check:
                    final_action = PlayerAction.BET
                    bet_value = max(min_bb, amount)
                    cost = min(bet_value, stack); final_amount = cost
                else: final_action = PlayerAction.FOLD
            else: # Valid raise context
                min_raise_delta = min_bb
                min_total_bet = current_bet + min_raise_delta
                min_action_cost = max(0, min_total_bet - my_bet_in_round) # Cost to reach min raise level
                actual_cost = max(min_action_cost, amount) # Ensure raise is at least minimal
                cost = min(actual_cost, stack); final_amount = cost
        elif action == PlayerAction.ALL_IN:
            cost = stack; final_amount = stack

        # Final check: if cost is effectively all-in
        if cost >= stack and stack > 0 and final_action not in [PlayerAction.CHECK, PlayerAction.FOLD]:
            final_action = PlayerAction.ALL_IN
            final_amount = stack
        # Ensure CHECK/FOLD if cost is 0 for other actions
        elif cost <= 0 and final_action not in [PlayerAction.CHECK, PlayerAction.FOLD]:
            final_action = PlayerAction.CHECK if can_check else PlayerAction.FOLD
            final_amount = 0

        return final_action, final_amount

    # --- MCTS Core ---
    def _monte_carlo_tree_search(self, game_state: list[int], num_players: int) -> Tuple[int, int]:
        """Runs MCTS."""
        initial_bets = {i: 0 for i in range(num_players)}
        initial_bets[self.my_index] = self.bet_amount # Start node reflects current reality

        root = MCTSNode(game_state, [], None, self.my_index, initial_bets, self.my_index)

        if root.game_state[12 + self.my_index] <= 0: return PlayerAction.CHECK, 0

        for _ in range(self.MCTS_SIMULATIONS):
            node = self._tree_policy(root, num_players)
            if node:
                reward = self._default_policy(node, num_players)
                self._backup(node, reward)

        best_child = self._best_child(root, 0) # Exploit

        if best_child and best_child.action is not None:
            return best_child.action, best_child.amount
        else: # Fallback
            call = max(0, root.game_state[8] - self.bet_amount)
            return (PlayerAction.CHECK, 0) if call == 0 else (PlayerAction.FOLD, 0)

    def _tree_policy(self, node: MCTSNode, num_p: int) -> Optional[MCTSNode]:
        """Selects / expands nodes."""
        cur = node
        while cur and not cur.is_terminal:
            stacks = cur.game_state[12:12+num_p]
            player_idx = cur.current_player_index
            if not (0 <= player_idx < num_p) or stacks[player_idx] <= 0: break # Player can't act

            if not cur.is_fully_expanded(num_p, self):
                return self._expand(cur, num_p)
            else:
                cur = self._best_child(cur, self.EXPLORATION_CONSTANT)
        return cur

    def _expand(self, node: MCTSNode, num_p: int) -> Optional[MCTSNode]:
        """Adds one child node."""
        player_idx = node.current_player_index
        stacks = node.game_state[12:12+num_p]
        stack = stacks[player_idx]

        possible = self._get_possible_actions(
            node.game_state[8], node.game_state[7], stack, node.game_state[9],
            node.player_bets_in_round.get(player_idx, 0)
        )

        unexplored = []
        for act, amt in possible:
            key = self._standardize_action_tuple(act, amt, stack)
            if key not in node.children: unexplored.append((act, amt))

        if not unexplored: return node # Should be fully expanded

        act, amt = random.choice(unexplored)
        key = self._standardize_action_tuple(act, amt, stack)

        # Simulate action (lightweight status)
        sim_status = {i: (PlayerStatus.ACTIVE if s > 0 else PlayerStatus.OUT) for i, s in enumerate(stacks)}
        next_state, next_bets, next_player, _ = self._simulate_mcts_action(
            node.game_state, node.player_bets_in_round, act, amt, player_idx, num_p, sim_status
        )

        child = MCTSNode(next_state, node.action_history + [(player_idx, act, amt)], node, node.ai_player_index, next_bets, next_player, act=act, amt=amt)
        node.children[key] = child
        return child

    def _standardize_action_tuple(self, action: int, amount: int, stack: int) -> Tuple[int, int]:
         """Standardizes action tuple, especially for ALL_IN."""
         amount = int(round(amount))
         stack = int(round(stack))
         # Determine effective cost based on action type (simplified)
         cost = 0
         if action == PlayerAction.ALL_IN: cost = stack
         elif action in [PlayerAction.BET, PlayerAction.RAISE, PlayerAction.CALL]: cost = amount

         if action not in [PlayerAction.FOLD, PlayerAction.CHECK] and cost >= stack and stack > 0:
              return (PlayerAction.ALL_IN, stack)
         return (action, amount)

    def _best_child(self, node: MCTSNode, exploration: float) -> Optional[MCTSNode]:
        """Selects best child via UCT."""
        if not node.children: return None
        log_parent_n = math.log(node.n + 1e-6)
        best_uct = -float('inf')
        best_node = None

        # Use items() for potentially better iteration if dicts grow large
        for child in node.children.values():
            if child.n == 0:
                uct = float('inf') # Prioritize unvisited
            else:
                uct = (child.q / child.n) + exploration * math.sqrt(log_parent_n / child.n)

            if uct > best_uct:
                best_uct = uct
                best_node = child

        return best_node if best_node else random.choice(list(node.children.values())) # Fallback


    def _default_policy(self, node: MCTSNode, num_p: int) -> float:
        """Fast, random simulation (rollout)."""
        state = list(node.game_state) # Shallow copy is faster if mutations are careful
        bets = dict(node.player_bets_in_round)
        cur_p = node.current_player_index
        stacks = state[12:12+num_p]
        status = {i: (PlayerStatus.ACTIVE if s > 0 else PlayerStatus.OUT) for i, s in enumerate(stacks)}

        for _ in range(self.MAX_MCTS_DEPTH * num_p): # Loop limit based on depth and players
             if self._is_terminal_state_sim(state, status): break
             if not (0 <= cur_p < num_p): break # Safety break

             player_stack = stacks[cur_p] # Direct access after ensuring index validity
             player_status = status.get(cur_p, PlayerStatus.OUT)

             if player_status != PlayerStatus.ACTIVE:
                 cur_p = self._find_next_acting_player(cur_p, num_p, status)
                 if cur_p == -1: break
                 continue # Skip turn

             player_bet = bets.get(cur_p, 0)
             poss_actions = self._get_possible_actions(state[8], state[7], player_stack, state[9], player_bet)

             if not poss_actions: # Forced check/fold
                 act = PlayerAction.CHECK if max(0, state[8] - player_bet) == 0 else PlayerAction.FOLD
                 amt = 0
             else:
                 act, amt = random.choice(poss_actions)

             # Simulate action (updates state, bets, status in-place if possible, else copies)
             state, bets, next_p, status = self._simulate_mcts_action(state, bets, act, amt, cur_p, num_p, status)
             stacks = state[12:12+num_p] # Re-sync stacks view
             cur_p = next_p

        return self._calculate_reward_sim(state, status, node.ai_player_index, num_p)

    def _backup(self, node: Optional[MCTSNode], reward: float) -> None:
        """Backpropagates reward."""
        cur = node
        while cur:
            cur.n += 1
            cur.q += reward
            cur = cur.parent

    # --- Simulation Helpers ---
    def _simulate_mcts_action(self, gs: list, bets: dict, act: int, amt: int, p_idx: int, num_p: int, p_status: Dict[int, int]) -> Tuple[list, dict, int, Dict[int, int]]:
        """Simulates action - MUTATES state for performance in rollout."""
        # NOTE: This mutates gs, bets, p_status. If using elsewhere, ensure deepcopy first.
        stacks = gs[12:12+num_p]
        pot = gs[7]
        cur_bet = gs[8]
        p_stack = stacks[p_idx]
        p_cur_bet = bets.get(p_idx, 0)
        cost, final_bet = 0, p_cur_bet
        new_high_bet = cur_bet

        if act == PlayerAction.FOLD: p_status[p_idx] = PlayerStatus.FOLDED
        elif act == PlayerAction.CALL: cost = min(max(0, cur_bet - p_cur_bet), p_stack); final_bet += cost
        elif act == PlayerAction.BET: cost = min(max(self._current_bb if self._current_bb > 0 else 1, amt), p_stack); final_bet += cost; new_high_bet = final_bet
        elif act == PlayerAction.RAISE: cost = min(amt, p_stack); final_bet += cost; new_high_bet = final_bet # Assume amt is valid cost
        elif act == PlayerAction.ALL_IN: cost = p_stack; final_bet += cost; new_high_bet = max(cur_bet, final_bet)

        if cost > 0: stacks[p_idx] -= cost; gs[7] = pot + cost
        bets[p_idx] = final_bet
        gs[8] = new_high_bet
        if stacks[p_idx] <= 0 and p_status[p_idx] != PlayerStatus.FOLDED: p_status[p_idx] = PlayerStatus.ALL_IN

        next_p = self._find_next_acting_player(p_idx, num_p, p_status)
        # Basic round end / card dealing simulation could be added here if needed
        # e.g., check if next_p indicates round end, deal cards to gs[2:7], reset gs[8], bets
        return gs, bets, next_p, p_status

    def _find_next_acting_player(self, cur_idx: int, num_p: int, p_status: Dict[int, int]) -> int:
        for i in range(1, num_p + 1):
            next_idx = (cur_idx + i) % num_p
            if p_status.get(next_idx) == PlayerStatus.ACTIVE: return next_idx
        return -1 # No active players

    def _is_terminal_state_sim(self, gs: list, p_status: Dict[int, int]) -> bool:
        num_p = gs[11]
        not_folded = sum(1 for i in range(num_p) if p_status.get(i, PlayerStatus.FOLDED) != PlayerStatus.FOLDED)
        if not_folded <= 1: return True
        community_count = sum(1 for i in gs[2:7] if i is not None and i >= -2) # Allow placeholders
        if community_count >= 5: return True # Assume terminal after river
        can_act = sum(1 for i in range(num_p) if p_status.get(i) == PlayerStatus.ACTIVE)
        if not_folded > 1 and can_act == 0: return True # All remaining are all-in
        return False

    def _calculate_reward_sim(self, term_gs: list, p_status: Dict[int, int], ai_idx: int, num_p: int) -> float:
        """Calculates 0/1 reward at simulation end."""
        if p_status.get(ai_idx) == PlayerStatus.FOLDED: return 0.0
        contenders = [i for i in range(num_p) if p_status.get(i, PlayerStatus.FOLDED) != PlayerStatus.FOLDED]
        if len(contenders) == 1: return 1.0 if contenders[0] == ai_idx else 0.0
        if not contenders or ai_idx not in contenders: return 0.0 # Should not happen if AI !folded

        stacks = term_gs[12:12+num_p]
        comm_indices = term_gs[2:7]
        comm_cards = [card_from_index(i) for i in comm_indices if i is not None and i >= 0]

        # Use stack comparison if ended before river showdown
        if len(comm_cards) < 5:
            ai_stack = stacks[ai_idx]
            max_opp_stack = max((stacks[i] for i in contenders if i != ai_idx), default=-1)
            return 1.0 if ai_stack >= max_opp_stack else 0.0

        # Crude Showdown
        try: ai_hand = HandEvaluator.evaluate_hand(self.hole_cards, comm_cards)
        except: ai_hand = HandResult(0, 0) # Default low hand on error

        # Simple check against "average" opponent hand (or just assume AI wins/loses randomly?)
        # The random opponent generation is slow and inaccurate. Let's simplify reward:
        # If AI reaches showdown, give a baseline reward (e.g., 0.5) or compare to a fixed hand?
        # For speed, let's just assume a 50/50 outcome if it goes to showdown in sim.
        # More advanced: use a quick equity estimation if available.
        # Sticking with previous crude comparison for now, but it's slow/inaccurate.
        best_opp_hand = HandResult(0,0)
        # (Keep the simple random opponent hand evaluation from previous version if needed,
        #  but acknowledge limitations. For pure speed optimization, replace with heuristic/random.)
        # --- Simplified random opponent eval (from previous code) ---
        used = {c.get_index() for c in self.hole_cards} | {c.get_index() for c in comm_cards}
        deck = [c for c in Deck().cards if c.get_index() not in used]
        random.shuffle(deck)
        for p_idx in contenders:
            if p_idx != ai_idx and len(deck) >= 2:
                 opp_hole = deck[:2]; deck = deck[2:]
                 try:
                      opp_res = HandEvaluator.evaluate_hand(opp_hole, comm_cards)
                      if opp_res > best_opp_hand: best_opp_hand = opp_res
                 except: pass # Ignore opponent eval errors
        # --- End simplified random opponent eval ---

        return 1.0 if ai_hand >= best_opp_hand else 0.0


    # --- Action Generation ---
    def _get_possible_actions(self, current_bet: int, pot_size: int, stack: int, bb: int, player_bet: int) -> List[Tuple[int, int]]:
         """Generates list of possible (Action, Amount) tuples."""
         if stack <= 0: return []
         call = max(0, current_bet - player_bet)
         can_check = (call == 0)
         possible = []
         min_bb = bb if bb > 0 else 1

         # Basic Actions
         if call > 0: possible.append((PlayerAction.FOLD, 0))
         if can_check: possible.append((PlayerAction.CHECK, 0))
         if call > 0:
             if call < stack: possible.append((PlayerAction.CALL, call))
             else: possible.append((PlayerAction.ALL_IN, stack)) # Call requires all-in

         # Betting/Raising (if stack allows more than just calling)
         if stack > call:
             # Add All-In if not already covered by CALL
             if not any(a == PlayerAction.ALL_IN for a,amt in possible):
                 possible.append((PlayerAction.ALL_IN, stack))

             # Add Bets (if checking is possible)
             if can_check:
                 min_bet = min(stack, min_bb)
                 bet_costs = [min_bet]
                 if pot_size > 0: bet_costs.extend([int(pot_size * 0.5), int(pot_size * 1.0)]) # Fewer sizes
                 for cost in sorted(list(set(bet_costs))):
                      if cost > 0 and cost < stack: possible.append((PlayerAction.BET, cost))

             # Add Raises (if calling is possible)
             if call > 0:
                 min_raise_delta = min_bb
                 min_total_bet = current_bet + min_raise_delta
                 min_raise_action_cost = max(0, min_total_bet - player_bet)

                 if stack >= min_raise_action_cost: # Can afford min raise
                      raise_costs = [min_raise_action_cost]
                      # Pot raise cost (simplified: call + ~pot size)
                      pot_raise_cost = call + pot_size + player_bet # Estimate pot after call
                      if pot_raise_cost > min_raise_action_cost: raise_costs.append(pot_raise_cost)

                      for cost in sorted(list(set(raise_costs))):
                          actual_cost = max(min_raise_action_cost, cost)
                          if actual_cost > 0 and actual_cost < stack:
                              possible.append((PlayerAction.RAISE, actual_cost))

         # Final filter for unique standardized actions
         final_set = set()
         result = []
         for action, amount in possible:
              key = self._standardize_action_tuple(action, amount, stack)
              if key not in final_set:
                 final_set.add(key)
                 result.append(key)
         return result

    # --- Win Tracking ---
    def update_winning_percentage(self, won: bool):
        self.total_games += 1
        if won: self.wins += 1
        self.winning_percentage = (self.wins / self.total_games) * 100 if self.total_games > 0 else 0.0

    def reset_game_stats(self):
        self.total_games = 0; self.wins = 0; self.winning_percentage = 0.0
        # Reset opponent models if they existed