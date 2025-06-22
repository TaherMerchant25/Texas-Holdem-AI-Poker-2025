from enum import Enum
from typing import List
from dataclasses import dataclass
import random


class Suit(Enum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass
class Card:
    rank: Rank
    suit: Suit

    def get_index(self) -> int:
        """
        Returns the index of the card rank. Indexing starts from 1.
        """
        return (self.suit.value * 13) + self.rank.value - 1
    
    def __str__(self) -> str:
        rank_symbols = {
            Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
            Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
            Rank.TEN: "10", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
        }
        suit_symbols = {
            Suit.HEARTS: "♥", Suit.DIAMONDS: "♦", Suit.CLUBS: "♣", Suit.SPADES: "♠"
        }
        return f"{rank_symbols[self.rank]}{suit_symbols[self.suit]}"


class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in Rank for suit in Suit]
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int = 1) -> List[Card]:
        """
        Deals num_cards cards from the deck, and discards them.
        """
        return [self.cards.pop() for _ in range(min(num_cards, len(self.cards)))]
