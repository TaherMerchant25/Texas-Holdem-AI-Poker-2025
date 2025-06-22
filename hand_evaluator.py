from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass
from collections import Counter
from card import Card


class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


@dataclass
class HandResult:
    hand_rank: HandRank
    hand_value: Tuple
    best_hand: List[Card]


class HandEvaluator:
    @staticmethod
    def evaluate_hand(player_cards: List[Card], community_cards: List[Card]) -> HandResult:
        all_cards = player_cards + community_cards
        
        # Check all 5-card combinations from the 7 cards
        best_hand_rank = HandRank.HIGH_CARD
        best_hand_value = (0,)
        best_hand = []
        
        # For a real engine, this would use a more efficient algorithm
        # Instead of checking all combinations
        from itertools import combinations
        for hand in combinations(all_cards, 5):
            hand_result = HandEvaluator._evaluate_five_card_hand(list(hand))
            
            if (hand_result.hand_rank.value > best_hand_rank.value or 
                (hand_result.hand_rank == best_hand_rank and hand_result.hand_value > best_hand_value)):
                best_hand_rank = hand_result.hand_rank
                best_hand_value = hand_result.hand_value
                best_hand = hand_result.best_hand
        
        return HandResult(best_hand_rank, best_hand_value, best_hand)
    
    @staticmethod
    def _evaluate_five_card_hand(hand: List[Card]) -> HandResult:
        # Count ranks and suits
        rank_counter = Counter([card.rank for card in hand])
        suit_counter = Counter([card.suit for card in hand])
        
        # Check for flush
        is_flush = max(suit_counter.values()) == 5
        
        # Check for straight
        ranks = sorted([card.rank.value for card in hand])
        is_straight = (len(set(ranks)) == 5 and max(ranks) - min(ranks) == 4)
        
        # Special case: A-5 straight (A counts as 1)
        if not is_straight and set(ranks) == {2, 3, 4, 5, 14}:
            is_straight = True
            ranks = [1, 2, 3, 4, 5]  # Treat Ace as 1 for this straight
        
        # Determine hand type and return result
        if is_straight and is_flush:
            if set(ranks) == {10, 11, 12, 13, 14}:
                return HandResult(HandRank.ROYAL_FLUSH, (10,), hand)
            return HandResult(HandRank.STRAIGHT_FLUSH, (max(ranks),), hand)
        
        if 4 in rank_counter.values():
            quads = [r.value for r, count in rank_counter.items() if count == 4][0]
            kicker = [r.value for r, count in rank_counter.items() if count == 1][0]
            return HandResult(HandRank.FOUR_OF_A_KIND, (quads, kicker), hand)
        
        if 3 in rank_counter.values() and 2 in rank_counter.values():
            trips = [r.value for r, count in rank_counter.items() if count == 3][0]
            pair = [r.value for r, count in rank_counter.items() if count == 2][0]
            return HandResult(HandRank.FULL_HOUSE, (trips, pair), hand)
        
        if is_flush:
            return HandResult(HandRank.FLUSH, tuple(sorted(ranks, reverse=True)), hand)
        
        if is_straight:
            return HandResult(HandRank.STRAIGHT, (max(ranks),), hand)
        
        if 3 in rank_counter.values():
            trips = [r.value for r, count in rank_counter.items() if count == 3][0]
            kickers = sorted([r.value for r, count in rank_counter.items() if count == 1], reverse=True)
            return HandResult(HandRank.THREE_OF_A_KIND, (trips, *kickers), hand)
        
        if list(rank_counter.values()).count(2) == 2:
            pairs = sorted([r.value for r, count in rank_counter.items() if count == 2], reverse=True)
            kicker = [r.value for r, count in rank_counter.items() if count == 1][0]
            return HandResult(HandRank.TWO_PAIR, (pairs[0], pairs[1], kicker), hand)
        
        if 2 in rank_counter.values():
            pair = [r.value for r, count in rank_counter.items() if count == 2][0]
            kickers = sorted([r.value for r, count in rank_counter.items() if count == 1], reverse=True)
            return HandResult(HandRank.PAIR, (pair, *kickers), hand)
        
        return HandResult(HandRank.HIGH_CARD, tuple(sorted(ranks, reverse=True)), hand)