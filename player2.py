# import random
# from collections import Counter
# from typing import List, Tuple, Optional
# import math
# import time
# import copy

# # Assuming card.py, hand_evaluator.py, game.py are accessible
# from card import Card, Rank, Suit, Deck
# from hand_evaluator import HandEvaluator, HandRank, HandResult
# from player import Player, PlayerAction, PlayerStatus
# from game import GamePhase

# # --- Helper function to get Card from index ---
# _FULL_DECK_FOR_INDEXING = Deck().cards
# def card_from_index(index: int) -> Optional[Card]:
#     """Maps an index (0-51) back to a Card object."""
#     if 0 <= index < 52:
#         for card in _FULL_DECK_FOR_INDEXING:
#              if card.get_index() == index:
#                  return card
#     return None
# # --- End Helper ---

# class MonteCarloGTOAIPlayer(Player):
#     """
#     An AI player that combines Monte Carlo Tree Search (MCTS) with
#     GTO-inspired elements and adaptive strategies for a very high win rate.
#     """

#     # --- MCTS Configuration ---
#     MCTS_SIMULATIONS = 250  # Number of Monte Carlo simulations per decision
#     EXPLORATION_CONSTANT = math.sqrt(2) # Exploration vs. Exploitation balance
#     MAX_MCTS_DEPTH = 5       # Limit MCTS depth for performance

#     # --- GTO & Adaptive Strategy Configuration ---
#     BASELINE_AGGRESSION = 0.7 # Overall aggression level
#     ADAPTATION_RATE = 0.05  # How quickly to adapt to opponent tendencies
#     BLUFF_THRESHOLD = 0.3  # Minimum probability for a bluff to be considered
#     VALUE_THRESHOLD = 0.7  # Minimum hand strength for a value bet/raise

#     # --- Opponent Modeling Configuration ---
#     HISTORY_LENGTH = 10      # Number of past actions to remember for opponent modeling
#     FOLD_BIAS = 0.9 # fold tendency = 0.5/ (1 - avg. freq) if > FOLD_BIAS fold more otherwise less fold


#     def __init__(self, name: str, stack: int):
#         super().__init__(name, stack)
#         self.winning_percentage = 0.0
#         self.total_games = 0
#         self.wins = 0
#         self.opponent_history = {}  # Store action history of opponents
#         self.opponent_fold_frequency = {} # Tracks how often opponents fold
#         self.opponent_bluffing_frequency = {}  # Tracks bluffing frequency
#         self.hand_strengths_history = []  # Store past hand strengths (postflop)
#         self.my_index = None  # Initialize my_index to None

#     def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
#         """
#         Determines the AI's action using a combination of MCTS, GTO principles,
#         and adaptive strategies based on opponent modeling.
#         """

#         start_time = time.time()

#         # --- 1. Parse Game State ---
#         hole_card_indices = game_state[0:2]
#         community_card_indices = game_state[2:7]
#         pot = game_state[7]
#         current_bet = game_state[8]  # The total amount players need to call
#         big_blind = game_state[9]
#         self.my_index = game_state[10] # Correctly assign my_index
#         num_players = game_state[11]
#         player_stacks = game_state[12:12 + num_players]

#         # --- Get actual Card objects ---
#         self.hole_cards = [card for i in hole_card_indices if (card := card_from_index(i))]
#         if not self.hole_cards or len(self.hole_cards) < 2:  # Safety check
#             print(f"AI {self.name} received invalid hole cards: {hole_card_indices}")
#             call_amount_safe = max(0, current_bet - self.bet_amount)
#             return (PlayerAction.FOLD, 0) if call_amount_safe > 0 else (PlayerAction.CHECK, 0)

#         community_cards = [card for i in community_card_indices if (card := card_from_index(i))]

#         # --- Basic Info ---
#         call_amount = max(0, current_bet - self.bet_amount)
#         can_check = (call_amount == 0)
#         phase = self._get_phase(community_cards)

#         # --- Store Big Blind for helpers ---
#         self._current_bb = big_blind  # Store BB temporarily for sizing functions

#         # --- Opponent Modeling ---
#         self._update_opponent_history(action_history, num_players)

#         # --- Monte Carlo Tree Search ---
#         best_action, best_amount = self._monte_carlo_tree_search(
#             game_state, action_history, num_players, community_cards, self.my_index
#         )
#         end_time = time.time()
#         print(f"MCTS took {end_time - start_time} seconds")

#         return best_action, best_amount

#     def _monte_carlo_tree_search(self, game_state: list[int], action_history: list, num_players: int, community_cards: List[Card], my_index: int) -> Tuple[PlayerAction, int]:
#         """
#         Implements the Monte Carlo Tree Search algorithm to determine the best action.
#         """

#         root = MCTSNode(game_state, action_history, None, my_index)
#         # Exploration Step, we see different game stats and situations based on our action
#         for _ in range(self.MCTS_SIMULATIONS):
#             node = self._tree_policy(root, num_players, community_cards, my_index)
#             reward = self._default_policy(node, num_players, community_cards, my_index)
#             self._backup(node, reward)

#         # Selection Step, now we choose which route is the best for our player
#         best_child = self._best_child(root, 0)  # 0 for exploitation
#         return best_child.action, best_child.amount

#     def _tree_policy(self, node: 'MCTSNode', num_players: int, community_cards: List[Card], my_index: int) -> 'MCTSNode':
#         """Selects the next node to explore using UCT."""
#         while not node.is_terminal:
#             if not node.is_fully_expanded(num_players, self.stack, self._current_bb, node.game_state[8], node.game_state[7], my_index): # Check if fully expanded using is_fully_expanded
#                 return self._expand(node, num_players, my_index, community_cards)
#             else:
#                 node = self._best_child(node, self.EXPLORATION_CONSTANT) # use exploration constant from UCT to select node.
#         return node


#     def _expand(self, node: 'MCTSNode', num_players: int, my_index: int, community_cards: List[Card]) -> 'MCTSNode':
#         """Expands the node by creating a new child node for an unexplored action."""
#         possible_actions = self._get_possible_actions(node.game_state, num_players, self.stack, self._current_bb, node.game_state[8], node.game_state[7], my_index) # Get actions using helper

#         # Filter already expanded actions
#         unexplored_actions = [(action, amount) for action, amount in possible_actions if (action, amount) not in node.children]

#         #Choose a random unexplored action
#         action, amount = random.choice(unexplored_actions)
#         next_game_state = self._simulate_action(node.game_state, action, amount, my_index, num_players)
#         child_node = MCTSNode(next_game_state, node.action_history + [(my_index, action, amount)], action, my_index, action=action, amount=amount)
#         node.children[(action, amount)] = child_node
#         return child_node


#     def _best_child(self, node: 'MCTSNode', exploration_constant: float) -> 'MCTSNode':
#         """Selects the best child node based on UCT value."""
#         best_child = None
#         best_uct = -float('inf')

#         for child in node.children.values():
#             uct_value = child.q / (child.n + 1e-6) + exploration_constant * math.sqrt(math.log(node.n + 1) / (child.n + 1e-6))
#             if uct_value > best_uct:
#                 best_uct = uct_value
#                 best_child = child

#         return best_child

#     def _default_policy(self, node: 'MCTSNode', num_players: int, community_cards: List[Card], my_index: int) -> float:
#         """Simulates a random game from the given node to a terminal state and returns the reward."""
#         current_game_state = copy.deepcopy(node.game_state)
#         current_action_history = copy.deepcopy(node.action_history)
#         current_player = my_index # start with the current player

#         depth = 0 # Add depth limit

#         while not self._is_terminal_state(current_game_state) and depth < self.MAX_MCTS_DEPTH:
#             possible_actions = self._get_possible_actions(current_game_state, num_players, self.stack, self._current_bb, current_game_state[8], current_game_state[7], current_player)

#             if not possible_actions: # no possible actions = fold
#                 return 0.0

#             action, amount = random.choice(possible_actions)
#             current_game_state = self._simulate_action(current_game_state, action, amount, current_player, num_players)
#             current_action_history.append((current_player, action, amount))

#             # progress to the next player using current_player rather than my_index
#             current_player = (current_player + 1) % num_players # progress to the next player

#             depth += 1 # Increment the depth

#         # Reward is 1 if we win, 0 otherwise.
#         reward = self._calculate_reward(current_game_state, my_index, num_players) # return the reward instead of printing
#         return reward

#     def _backup(self, node: 'MCTSNode', reward: float) -> None:
#         """Updates the node statistics along the path from the expanded node to the root."""
#         while node is not None:
#             node.n += 1
#             node.q += reward
#             node = node.parent # traverse back to the root node

#     def _is_terminal_state(self, game_state: list[int]) -> bool:
#         """Checks if the given game state is a terminal state."""
#         # Simplified terminal state check (adjust as needed)
#         player_stacks = game_state[12:]
#         active_players = sum(1 for stack in player_stacks if stack > 0)
#         community_cards = game_state[2:7]

#         # if there are 1 or less active players, then game is over
#         if active_players <= 1:
#             return True

#         # If the River is out, and all players have acted game is over
#         if len([c for c in community_cards if c > 0]) == 5:
#              return True
#         return False

#     def _calculate_reward(self, game_state: list[int], my_index: int, num_players: int) -> float:
#          """Calculates the reward (win/loss) for the AI in the terminal state."""
#          player_stacks = game_state[12:]
#          my_stack = player_stacks[my_index]

#          # reward is 1 if our player has the most chips at the end
#          winning_stack = max(player_stacks) # find max stack from player_stacks
#          if my_stack == winning_stack:
#               return 1.0
#          else:
#               return 0.0


#     def _get_possible_actions(self, game_state: list[int], num_players: int, stack_size: int, big_blind: int, current_bet: int, pot_size: int, my_index: int) -> List[Tuple[PlayerAction, int]]:
#          """Determines all possible actions for the AI in the current state."""
#          player_stacks = game_state[12:]

#          my_stack = player_stacks[my_index]

#          call_amount = max(0, current_bet - self.bet_amount)
#          can_check = (call_amount == 0)

#          possible_actions = []

#          # if player can check
#          if can_check:
#               possible_actions.append((PlayerAction.CHECK, 0))
#               bet_amount = int(min(stack_size, pot_size * (0.5 + (0.2 * self.BASELINE_AGGRESSION)))) # Bet between 50%-70% of the pot
#               bet_amount = max(bet_amount, big_blind) # Min bet is big blind
#               if bet_amount > 0 : possible_actions.append((PlayerAction.BET, bet_amount))
#          else:
#               # if player can call
#               if call_amount <= stack_size:
#                    possible_actions.append((PlayerAction.CALL, call_amount))
#               #player can fold
#               possible_actions.append((PlayerAction.FOLD, 0))

#          # Player can ALL_IN if they have chips
#          if stack_size > 0:
#               possible_actions.append((PlayerAction.ALL_IN, stack_size))

#          # Player can RAISE if they have chips and can afford the min raise
#          min_raise_amount = big_blind # 1 big blind

#          if call_amount < stack_size: # Only add raise if the player can afford to call
#             raise_amount = int(min(stack_size - call_amount, pot_size * (0.7 + (0.3 * self.BASELINE_AGGRESSION)))) # Raise between 70% - 100% of pot

#          return possible_actions

#     def _simulate_action(self, game_state: list[int], action: PlayerAction, amount: int, acting_player_index: int, num_players: int) -> list[int]:
#          """Simulates an action and returns the updated game state."""

#          new_game_state = copy.deepcopy(game_state) # create a new copy so we don't mess up the tree

#          player_stacks = new_game_state[12:]
#          community_cards = new_game_state[2:7]

#          # Adjust pot and player stacks based on the action
#          if action == PlayerAction.BET or action == PlayerAction.RAISE:
#               player_stacks[acting_player_index] -= amount
#               new_game_state[7] += amount # pot size increases

#          elif action == PlayerAction.CALL:
#               call_amount = max(0, new_game_state[8] - self.bet_amount)
#               player_stacks[acting_player_index] -= call_amount
#               new_game_state[7] += call_amount

#          elif action == PlayerAction.ALL_IN:
#               new_game_state[7] += player_stacks[acting_player_index] # all chips go into the pot
#               player_stacks[acting_player_index] = 0 # player has no more chips

#          # Update game state with changed player stacks
#          new_game_state[12:] = player_stacks

#          return new_game_state


#     def _get_phase(self, community_cards: List[Card]) -> GamePhase:
#         """Determines the current game phase."""
#         num_community = len(community_cards)
#         if num_community == 0: return GamePhase.PRE_FLOP
#         elif num_community == 3: return GamePhase.FLOP
#         elif num_community == 4: return GamePhase.TURN
#         elif num_community == 5: return GamePhase.RIVER
#         else: return GamePhase.SETUP

#     def _get_effective_position(self, my_index: int, num_players: int, action_history: list, phase: GamePhase) -> str:
#         """Approximates position. Crude without button knowledge."""
#         relative_pos = my_index / float(num_players) if num_players > 0 else 0
#         if relative_pos < 0.33: return "early"
#         if relative_pos < 0.66: return "middle"
#         return "late"

#     def _classify_preflop_hand(self) -> str:
#         """ Classifies starting hand strength. """
#         if not self.hole_cards or len(self.hole_cards) != 2: return "weak"
#         c1, c2 = sorted(self.hole_cards, key=lambda c: c.rank.value, reverse=True)
#         r1, r2 = c1.rank, c2.rank
#         suited = (c1.suit == c2.suit)
#         vals = sorted([r1.value, r2.value], reverse=True)
#         is_pair = (vals[0] == vals[1])

#         if is_pair and vals[0] >= Rank.JACK.value: return "premium"
#         if vals == [Rank.ACE.value, Rank.KING.value]: return "premium"
#         if is_pair and vals[0] >= Rank.NINE.value : return "strong"
#         if suited and vals[0] == Rank.ACE.value and vals[1] >= Rank.JACK.value: return "strong"
#         if suited and vals == [Rank.KING.value, Rank.QUEEN.value]: return "strong"
#         if not suited and vals == [Rank.ACE.value, Rank.QUEEN.value]: return "strong"
#         if is_pair: return "good"
#         if suited:
#             if vals[0] == Rank.ACE.value: return "good"
#             if vals[0] == Rank.KING.value and vals[1] >= Rank.TEN.value: return "good"
#             if vals[0] == Rank.QUEEN.value and vals[1] >= Rank.TEN.value: return "good"
#             if vals[0] == Rank.JACK.value and vals[1] >= Rank.NINE.value: return "good"
#             if vals[0] == Rank.TEN.value and vals[1] >= Rank.NINE.value: return "good"
#             if vals[0] == Rank.NINE.value and vals[1] >= Rank.EIGHT.value: return "good"
#             if vals[0] == Rank.EIGHT.value and vals[1] >= Rank.SEVEN.value: return "good"
#             if vals[0] == Rank.SEVEN.value and vals[1] >= Rank.SIX.value: return "good"
#         if not suited:
#              if vals[0] == Rank.ACE.value and vals[1] >= Rank.TEN.value: return "good"
#              if vals == [Rank.KING.value, Rank.QUEEN.value]: return "strong"
#         if suited and vals[1] >= Rank.FIVE.value and (vals[0] - vals[1] <= 2): return "speculative"
#         if not suited and vals[0] == Rank.KING.value and vals[1] == Rank.JACK.value: return "speculative"
#         return "weak"

#     def _evaluate_postflop_hand(self, community_cards: List[Card]) -> Tuple[HandResult, Optional[str], int]:
#         """Evaluates hand strength, identifies draws, and counts outs."""
#         if not self.hole_cards: return HandEvaluator.evaluate_hand([], community_cards), None, 0
#         best_eval = HandEvaluator.evaluate_hand(self.hole_cards, community_cards)
#         return best_eval, None, 0 # Simplified for MCTS

#     def _calculate_pot_odds_percentage(self, pot: int, call_amount: int) -> float:
#         """Calculates pot odds as a percentage required equity."""
#         if call_amount <= 0: return 0.0
#         total_pot_after_call = pot + call_amount + call_amount
#         if total_pot_after_call == 0: return 100.0
#         return (call_amount / total_pot_after_call) * 100

#     def _update_opponent_history(self, action_history: list, num_players: int) -> None:
#         """Updates the action history and fold frequency for each opponent."""
#         for player_index in range(num_players):
#             if player_index != self.my_index: # only track opponents
#                 if player_index not in self.opponent_history:
#                     self.opponent_history[player_index] = []

#                 # Extract actions from history and update
#                 player_actions = []
#                 for item in action_history:
#                     if len(item) == 3:  # Check if the item has 3 elements
#                         p, a, b = item
#                         if p == player_index:
#                             player_actions.append((p, a, b))
#                     else:
#                         print(f"Unexpected action history format: {item}")


#                 self.opponent_history[player_index].extend(player_actions)

#                 # Keep limited history length
#                 self.opponent_history[player_index] = self.opponent_history[player_index][-self.HISTORY_LENGTH:]

#     def _get_opponent_fold_frequency(self, player_index: int) -> float:
#         """Calculates how often an opponent folds based on their history."""
#         if player_index not in self.opponent_history or not self.opponent_history[player_index]:
#             return 0.0

#         fold_count = sum(1 for _, action, _ in self.opponent_history[player_index] if action == PlayerAction.FOLD)
#         return fold_count / len(self.opponent_history[player_index])

#     def _adjust_bet_sizing(self, bet_amount: int, opponent_fold_frequency: float) -> int:
#         """Adjust bet sizing based on opponent fold frequency."""
#         # If opponent folds a lot, bet slightly larger to exploit that.
#         if opponent_fold_frequency > 0.5: # Folds more than 50% of the time
#             bet_amount = int(bet_amount * (1 + self.ADAPTATION_RATE))
#         return bet_amount

#     def _adjust_aggression_based_on_folds(self, opponent_index: int, can_check: bool) -> float:
#         """Adjusts the aggression level based on how often the opponent folds."""
#         fold_freq = self._get_opponent_fold_frequency(opponent_index)
#         if fold_freq > self.FOLD_BIAS:
#           #opponent is folding a lot, decrease our aggression, as we want action
#           return 0.5 / (1 - fold_freq) # linear decrease in aggression based on fold increase (y = 0.5/ (1- x))
#         else:
#           #otherwise maintain default aggression
#           return self.BASELINE_AGGRESSION

#     def update_winning_percentage(self, won: bool):
#         """Updates the AI's winning percentage."""
#         self.total_games += 1
#         self.wins += 1 if won else 0
#         self.winning_percentage = (self.wins / self.total_games) * 100 if self.total_games > 0 else 0

#     def reset_game_stats(self):
#         """Resets the game statistics for a new tournament."""
#         self.total_games = 0
#         self.wins = 0
#         self.winning_percentage = 0.0

# class MCTSNode:
#     """Represents a node in the Monte Carlo search tree."""
#     def __init__(self, game_state: list[int], action_history: list, parent: Optional['MCTSNode'], my_index : int, action: Optional[PlayerAction] = None, amount: int = 0):
#         self.game_state = game_state
#         self.action_history = action_history
#         self.parent = parent
#         self.children = {}  # Map from (action, amount) to MCTSNode
#         self.n = 0          # Number of visits
#         self.q = 0          # Total reward
#         self.action = action # action taken
#         self.amount = amount
#         self.my_index = my_index

#     @property
#     def is_terminal(self) -> bool:
#         """Checks if this node represents a terminal state."""
#         player_stacks = self.game_state[12:]
#         active_players = sum(1 for stack in player_stacks if stack > 0)
#         community_cards = self.game_state[2:7]
#         if active_players <= 1:
#             return True
#         if len([c for c in community_cards if c > 0]) == 5:
#              return True
#         return False

#     def is_fully_expanded(self, num_players: int, stack_size: int, big_blind: int, current_bet: int, pot_size: int, my_index : int) -> bool:
#         """Checks if all possible actions from this node have been explored."""
#         #Get all actions that are possible
#         possible_actions = MonteCarloGTOAIPlayer()._get_possible_actions(self.game_state, num_players, stack_size, big_blind, current_bet, pot_size, my_index)

#         #Check if possible actions have already been added to children
#         for action, amount in possible_actions:
#              if (action, amount) not in self.children:
#                   return False
#         return True # If all possible actions are in children then return True
import random
from collections import Counter
from typing import List, Tuple, Optional
import math
import time
import copy

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

class MonteCarloGTOAIPlayer(Player):
    """
    An AI player that combines Monte Carlo Tree Search (MCTS) with
    GTO-inspired elements and adaptive strategies for a very high win rate.
    """

    # --- MCTS Configuration ---
    MCTS_SIMULATIONS = 250  # Number of Monte Carlo simulations per decision
    EXPLORATION_CONSTANT = math.sqrt(2) # Exploration vs. Exploitation balance
    MAX_MCTS_DEPTH = 5       # Limit MCTS depth for performance

    # --- GTO & Adaptive Strategy Configuration ---
    BASELINE_AGGRESSION = 0.7 # Overall aggression level
    ADAPTATION_RATE = 0.05  # How quickly to adapt to opponent tendencies
    BLUFF_THRESHOLD = 0.3  # Minimum probability for a bluff to be considered
    VALUE_THRESHOLD = 0.7  # Minimum hand strength for a value bet/raise

    # --- Opponent Modeling Configuration ---
    HISTORY_LENGTH = 10      # Number of past actions to remember for opponent modeling
    FOLD_BIAS = 0.9 # fold tendency = 0.5/ (1 - avg. freq) if > FOLD_BIAS fold more otherwise less fold


    def __init__(self, name: str, stack: int):
        super().__init__(name, stack)
        self.winning_percentage = 0.0
        self.total_games = 0
        self.wins = 0
        self.opponent_history = {}  # Store action history of opponents
        self.opponent_fold_frequency = {} # Tracks how often opponents fold
        self.opponent_bluffing_frequency = {}  # Tracks bluffing frequency
        self.hand_strengths_history = []  # Store past hand strengths (postflop)
        self.my_index = None  # Initialize my_index to None

    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
        """
        Determines the AI's action using a combination of MCTS, GTO principles,
        and adaptive strategies based on opponent modeling.
        """

        start_time = time.time()

        # --- 1. Parse Game State ---
        hole_card_indices = game_state[0:2]
        community_card_indices = game_state[2:7]
        pot = game_state[7]
        current_bet = game_state[8]  # The total amount players need to call
        big_blind = game_state[9]
        self.my_index = game_state[10] # Correctly assign my_index
        num_players = game_state[11]
        player_stacks = game_state[12:12 + num_players]

        # --- Get actual Card objects ---
        self.hole_cards = [card for i in hole_card_indices if (card := card_from_index(i))]
        if not self.hole_cards or len(self.hole_cards) < 2:  # Safety check
            print(f"AI {self.name} received invalid hole cards: {hole_card_indices}")
            call_amount_safe = max(0, current_bet - self.bet_amount)
            return (PlayerAction.FOLD, 0) if call_amount_safe > 0 else (PlayerAction.CHECK, 0)

        community_cards = [card for i in community_card_indices if (card := card_from_index(i))]

        # --- Basic Info ---
        call_amount = max(0, current_bet - self.bet_amount)
        can_check = (call_amount == 0)
        phase = self._get_phase(community_cards)

        # --- Store Big Blind for helpers ---
        self._current_bb = big_blind  # Store BB temporarily for sizing functions

        # --- Opponent Modeling ---
        self._update_opponent_history(action_history, num_players)

        # --- Monte Carlo Tree Search ---
        best_action, best_amount = self._monte_carlo_tree_search(
            game_state, action_history, num_players, community_cards, self.my_index
        )
        end_time = time.time()
        print(f"MCTS took {end_time - start_time} seconds")
        print(f"{self.name} Winning Percentage: {self.winning_percentage:.2f}%")
        return best_action, best_amount

    def _monte_carlo_tree_search(self, game_state: list[int], action_history: list, num_players: int, community_cards: List[Card], my_index: int) -> Tuple[PlayerAction, int]:
        """
        Implements the Monte Carlo Tree Search algorithm to determine the best action.
        """

        root = MCTSNode(game_state, action_history, None, my_index)
        # Exploration Step, we see different game stats and situations based on our action
        for _ in range(self.MCTS_SIMULATIONS):
            node = self._tree_policy(root, num_players, community_cards, my_index)
            reward = self._default_policy(node, num_players, community_cards, my_index)
            self._backup(node, reward)

        # Selection Step, now we choose which route is the best for our player
        best_child = self._best_child(root, 0)  # 0 for exploitation
        return best_child.action, best_child.amount

    def _tree_policy(self, node: 'MCTSNode', num_players: int, community_cards: List[Card], my_index: int) -> 'MCTSNode':
        """Selects the next node to explore using UCT."""
        while not node.is_terminal:
            if not node.is_fully_expanded(num_players, self.stack, self._current_bb, node.game_state[8], node.game_state[7], my_index): # Check if fully expanded using is_fully_expanded
                return self._expand(node, num_players, my_index, community_cards)
            else:
                node = self._best_child(node, self.EXPLORATION_CONSTANT) # use exploration constant from UCT to select node.
        return node


    def _expand(self, node: 'MCTSNode', num_players: int, my_index: int, community_cards: List[Card]) -> 'MCTSNode':
        """Expands the node by creating a new child node for an unexplored action."""
        possible_actions = self._get_possible_actions(node.game_state, num_players, self.stack, self._current_bb, node.game_state[8], node.game_state[7], my_index) # Get actions using helper

        # Filter already expanded actions
        unexplored_actions = [(action, amount) for action, amount in possible_actions if (action, amount) not in node.children]

        #Choose a random unexplored action
        action, amount = random.choice(unexplored_actions)
        next_game_state = self._simulate_action(node.game_state, action, amount, my_index, num_players)
        child_node = MCTSNode(next_game_state, node.action_history + [(my_index, action, amount)], action, my_index, action=action, amount=amount)
        node.children[(action, amount)] = child_node
        return child_node


    def _best_child(self, node: 'MCTSNode', exploration_constant: float) -> 'MCTSNode':
        """Selects the best child node based on UCT value."""
        best_child = None
        best_uct = -float('inf')

        for child in node.children.values():
            uct_value = child.q / (child.n + 1e-6) + exploration_constant * math.sqrt(math.log(node.n + 1) / (child.n + 1e-6))
            if uct_value > best_uct:
                best_uct = uct_value
                best_child = child

        return best_child

    def _default_policy(self, node: 'MCTSNode', num_players: int, community_cards: List[Card], my_index: int) -> float:
        """Simulates a random game from the given node to a terminal state and returns the reward."""
        current_game_state = copy.deepcopy(node.game_state)
        current_action_history = copy.deepcopy(node.action_history)
        current_player = my_index # start with the current player

        depth = 0 # Add depth limit

        while not self._is_terminal_state(current_game_state) and depth < self.MAX_MCTS_DEPTH:
            possible_actions = self._get_possible_actions(current_game_state, num_players, self.stack, self._current_bb, current_game_state[8], current_game_state[7], current_player)

            if not possible_actions: # no possible actions = fold
                return 0.0

            action, amount = random.choice(possible_actions)
            current_game_state = self._simulate_action(current_game_state, action, amount, current_player, num_players)
            current_action_history.append((current_player, action, amount))

            # progress to the next player using current_player rather than my_index
            current_player = (current_player + 1) % num_players # progress to the next player

            depth += 1 # Increment the depth

        # Reward is 1 if we win, 0 otherwise.
        reward = self._calculate_reward(current_game_state, my_index, num_players) # return the reward instead of printing
        return reward

    def _backup(self, node: 'MCTSNode', reward: float) -> None:
        """Updates the node statistics along the path from the expanded node to the root."""
        while node is not None:
            node.n += 1
            node.q += reward
            node = node.parent # traverse back to the root node

    def _is_terminal_state(self, game_state: list[int]) -> bool:
        """Checks if the given game state is a terminal state."""
        # Simplified terminal state check (adjust as needed)
        player_stacks = game_state[12:]
        active_players = sum(1 for stack in player_stacks if stack > 0)
        community_cards = game_state[2:7]

        # if there are 1 or less active players, then game is over
        if active_players <= 1:
            return True

        # If the River is out, and all players have acted game is over
        if len([c for c in community_cards if c > 0]) == 5:
             return True
        return False

    def _calculate_reward(self, game_state: list[int], my_index: int, num_players: int) -> float:
         """Calculates the reward (win/loss) for the AI in the terminal state."""
         player_stacks = game_state[12:]
         my_stack = player_stacks[my_index]

         # reward is 1 if our player has the most chips at the end
         winning_stack = max(player_stacks) # find max stack from player_stacks
         if my_stack == winning_stack:
              return 1.0
         else:
              return 0.0


    def _get_possible_actions(self, game_state: list[int], num_players: int, stack_size: int, big_blind: int, current_bet: int, pot_size: int, my_index: int) -> List[Tuple[PlayerAction, int]]:
         """Determines all possible actions for the AI in the current state."""
         player_stacks = game_state[12:]

         my_stack = player_stacks[my_index]

         call_amount = max(0, current_bet - self.bet_amount)
         can_check = (call_amount == 0)

         possible_actions = []

         # if player can check
         if can_check:
              possible_actions.append((PlayerAction.CHECK, 0))
              bet_amount = int(min(stack_size, pot_size * (0.5 + (0.2 * self.BASELINE_AGGRESSION)))) # Bet between 50%-70% of the pot
              bet_amount = max(bet_amount, big_blind) # Min bet is big blind
              if bet_amount > 0 : possible_actions.append((PlayerAction.BET, bet_amount))
         else:
              # if player can call
              if call_amount <= stack_size:
                   possible_actions.append((PlayerAction.CALL, call_amount))
              #player can fold
              possible_actions.append((PlayerAction.FOLD, 0))

         # Player can ALL_IN if they have chips
         if stack_size > 0:
              possible_actions.append((PlayerAction.ALL_IN, stack_size))

         # Player can RAISE if they have chips and can afford the min raise
         min_raise_amount = big_blind # 1 big blind

         if call_amount < stack_size: # Only add raise if the player can afford to call
            raise_amount = int(min(stack_size - call_amount, pot_size * (0.7 + (0.3 * self.BASELINE_AGGRESSION)))) # Raise between 70% - 100% of pot

         return possible_actions

    def _simulate_action(self, game_state: list[int], action: PlayerAction, amount: int, acting_player_index: int, num_players: int) -> list[int]:
         """Simulates an action and returns the updated game state."""

         new_game_state = copy.deepcopy(game_state) # create a new copy so we don't mess up the tree

         player_stacks = new_game_state[12:]
         community_cards = new_game_state[2:7]

         # Adjust pot and player stacks based on the action
         if action == PlayerAction.BET or action == PlayerAction.RAISE:
              player_stacks[acting_player_index] -= amount
              new_game_state[7] += amount # pot size increases

         elif action == PlayerAction.CALL:
              call_amount = max(0, new_game_state[8] - self.bet_amount)
              player_stacks[acting_player_index] -= call_amount
              new_game_state[7] += call_amount

         elif action == PlayerAction.ALL_IN:
              new_game_state[7] += player_stacks[acting_player_index] # all chips go into the pot
              player_stacks[acting_player_index] = 0 # player has no more chips

         # Update game state with changed player stacks
         new_game_state[12:] = player_stacks

         return new_game_state


    def _get_phase(self, community_cards: List[Card]) -> GamePhase:
        """Determines the current game phase."""
        num_community = len(community_cards)
        if num_community == 0: return GamePhase.PRE_FLOP
        elif num_community == 3: return GamePhase.FLOP
        elif num_community == 4: return GamePhase.TURN
        elif num_community == 5: return GamePhase.RIVER
        else: return GamePhase.SETUP

    def _get_effective_position(self, my_index: int, num_players: int, action_history: list, phase: GamePhase) -> str:
        """Approximates position. Crude without button knowledge."""
        relative_pos = my_index / float(num_players) if num_players > 0 else 0
        if relative_pos < 0.33: return "early"
        if relative_pos < 0.66: return "middle"
        return "late"

    def _classify_preflop_hand(self) -> str:
        """ Classifies starting hand strength. """
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
             if vals == [Rank.KING.value, Rank.QUEEN.value]: return "strong"
        if suited and vals[1] >= Rank.FIVE.value and (vals[0] - vals[1] <= 2): return "speculative"
        if not suited and vals[0] == Rank.KING.value and vals[1] == Rank.JACK.value: return "speculative"
        return "weak"

    def _evaluate_postflop_hand(self, community_cards: List[Card]) -> Tuple[HandResult, Optional[str], int]:
        """Evaluates hand strength, identifies draws, and counts outs."""
        if not self.hole_cards: return HandEvaluator.evaluate_hand([], community_cards), None, 0
        best_eval = HandEvaluator.evaluate_hand(self.hole_cards, community_cards)
        return best_eval, None, 0 # Simplified for MCTS

    def _calculate_pot_odds_percentage(self, pot: int, call_amount: int) -> float:
        """Calculates pot odds as a percentage required equity."""
        if call_amount <= 0: return 0.0
        total_pot_after_call = pot + call_amount + call_amount
        if total_pot_after_call == 0: return 100.0
        return (call_amount / total_pot_after_call) * 100

    def _update_opponent_history(self, action_history: list, num_players: int) -> None:
        """Updates the action history and fold frequency for each opponent."""
        for player_index in range(num_players):
            if player_index != self.my_index: # only track opponents
                if player_index not in self.opponent_history:
                    self.opponent_history[player_index] = []

                # Extract actions from history and update
                player_actions = []
                for item in action_history:
                    if len(item) == 3:  # Check if the item has 3 elements
                        p, a, b = item
                        if p == player_index:
                            player_actions.append((p, a, b))
                    else:
                        print(f"Unexpected action history format: {item}")


                self.opponent_history[player_index].extend(player_actions)

                # Keep limited history length
                self.opponent_history[player_index] = self.opponent_history[player_index][-self.HISTORY_LENGTH:]

    def _get_opponent_fold_frequency(self, player_index: int) -> float:
        """Calculates how often an opponent folds based on their history."""
        if player_index not in self.opponent_history or not self.opponent_history[player_index]:
            return 0.0

        fold_count = sum(1 for _, action, _ in self.opponent_history[player_index] if action == PlayerAction.FOLD)
        return fold_count / len(self.opponent_history[player_index])

    def _adjust_bet_sizing(self, bet_amount: int, opponent_fold_frequency: float) -> int:
        """Adjust bet sizing based on opponent fold frequency."""
        # If opponent folds a lot, bet slightly larger to exploit that.
        if opponent_fold_frequency > 0.5: # Folds more than 50% of the time
            bet_amount = int(bet_amount * (1 + self.ADAPTATION_RATE))
        return bet_amount

    def _adjust_aggression_based_on_folds(self, opponent_index: int, can_check: bool) -> float:
        """Adjusts the aggression level based on how often the opponent folds."""
        fold_freq = self._get_opponent_fold_frequency(opponent_index)
        if fold_freq > self.FOLD_BIAS:
          #opponent is folding a lot, decrease our aggression, as we want action
          return 0.5 / (1 - fold_freq) # linear decrease in aggression based on fold increase (y = 0.5/ (1- x))
        else:
          #otherwise maintain default aggression
          return self.BASELINE_AGGRESSION

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

class MCTSNode:
    """Represents a node in the Monte Carlo search tree."""
    def __init__(self, game_state: list[int], action_history: list, parent: Optional['MCTSNode'], my_index : int, action: Optional[PlayerAction] = None, amount: int = 0):
        self.game_state = game_state
        self.action_history = action_history
        self.parent = parent
        self.children = {}  # Map from (action, amount) to MCTSNode
        self.n = 0          # Number of visits
        self.q = 0          # Total reward
        self.action = action # action taken
        self.amount = amount
        self.my_index = my_index

    @property
    def is_terminal(self) -> bool:
        """Checks if this node represents a terminal state."""
        player_stacks = self.game_state[12:]
        active_players = sum(1 for stack in player_stacks if stack > 0)
        community_cards = self.game_state[2:7]
        if active_players <= 1:
            return True
        if len([c for c in community_cards if c > 0]) == 5:
             return True
        return False

    def is_fully_expanded(self, num_players: int, stack_size: int, big_blind: int, current_bet: int, pot_size: int, my_index : int) -> bool:
        """Checks if all possible actions from this node have been explored."""
        #Get all actions that are possible
        possible_actions = MonteCarloGTOAIPlayer()._get_possible_actions(self.game_state, num_players, stack_size, big_blind, current_bet, pot_size, my_index)

        #Check if possible actions have already been added to children
        for action, amount in possible_actions:
             if (action, amount) not in self.children:
                  return False
        return True # If all possible actions are in children then return True
