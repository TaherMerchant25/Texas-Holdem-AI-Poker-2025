from card import Suit, Rank, Card
from player import Player, PlayerAction
import pandas

"""
dataset can be found at: https://www.kaggle.com/datasets/hosseinah1/poker-game-dataset
"""

class StatPlayer(Player):

    def __init__(self, name, stack):
        super().__init__(name, stack)
        names = ["S1", "R1", "S2", "R2", "S3", "R3", "S4", "R4", "S5", "R5", "Score"]
        self.df = pandas.read_csv("poker-hand-testing.csv", header=0, names=names)

    def action(self, game_state: list[int], action_history: list):
        comm_cards = game_state[2:7]
        num_unrevealed = comm_cards.count(0)

        if num_unrevealed == 5:
            return self.pre_flop_strategy(game_state)
        elif num_unrevealed == 2:
            return self.flop_strategy(game_state)
        elif num_unrevealed == 1:
            return self.turn_strategy(game_state)
        else:
            return self.river_strategy(game_state)

    def pre_flop_strategy(self, game_state):
        card1 = game_state[0]
        card2 = game_state[1]

        suit1, rank1 = self.get_suit_rank(card1)
        suit2, rank2 = self.get_suit_rank(card2)

        score = self.df.loc[
            (self.df["S1"] == suit1) & (self.df["R1"] == rank1) & (self.df["S2"] == suit2) & (self.df["R2"] == rank2),
            "Score"
        ].mean()

        print(score)
        if score < 0.5:
            return PlayerAction.FOLD, 0
        return PlayerAction.CALL, 0

    def flop_strategy(self, game_state):
        card1 = game_state[0]
        card2 = game_state[1]

        suit1, rank1 = self.get_suit_rank(card1)
        suit2, rank2 = self.get_suit_rank(card2)

        best_score = 0
        comm_cards = game_state[2:5]  # first 3 cards are revealed
        for card in comm_cards:
            suit3, rank3 = self.get_suit_rank(card)
            score = self.df.loc[
                (self.df["S1"] == suit1) & (self.df["R1"] == rank1) & (self.df["S2"] == suit2) & (self.df["R2"] == rank2)
                & (self.df["S3"] == suit3) & (self.df["R3"] == rank3),
                "Score"
            ].mean()
            best_score = max(best_score, score)

        print(best_score)
        if best_score < 0.7:
            return PlayerAction.FOLD, 0
        return PlayerAction.CALL, 0

    def turn_strategy(self, game_state):
        card1 = game_state[0]
        card2 = game_state[1]

        suit1, rank1 = self.get_suit_rank(card1)
        suit2, rank2 = self.get_suit_rank(card2)

        best_score = 0
        comm_cards = game_state[2:6]  # first 4 cards are revealed
        for i in range(3):
            suit3, rank3 = self.get_suit_rank(comm_cards[i])
            for j in range(i + 1, 4):
                suit4, rank4 = self.get_suit_rank(comm_cards[j])
                score = self.df.loc[
                    (self.df["S1"] == suit1) & (self.df["R1"] == rank1) & (self.df["S2"] == suit2) & (self.df["R2"] == rank2)
                    & (self.df["S3"] == suit3) & (self.df["R3"] == rank3) & (self.df["S4"] == suit4) & (self.df["R4"] == rank4),
                    "Score"
                ].mean()
                best_score = max(best_score, score)

        print(best_score)
        if best_score < 1.0:
            return PlayerAction.FOLD, 0
        return PlayerAction.CALL, 0

    def river_strategy(self, game_state):
        card1 = game_state[0]
        card2 = game_state[1]

        suit1, rank1 = self.get_suit_rank(card1)
        suit2, rank2 = self.get_suit_rank(card2)

        best_score = 0
        comm_cards = game_state[2:7]  # first 4 cards are revealed
        for i in range(3):
            suit3, rank3 = self.get_suit_rank(comm_cards[i])
            for j in range(i + 1, 4):
                suit4, rank4 = self.get_suit_rank(comm_cards[j])
                for k in range(j + 1, 5):
                    suit5, rank5 = self.get_suit_rank(comm_cards[k])
                    score = self.df.loc[
                        (self.df["S1"] == suit1) & (self.df["R1"] == rank1) & (self.df["S2"] == suit2) &
                        (self.df["R2"] == rank2) & (self.df["S3"] == suit3) & (self.df["R3"] == rank3) &
                        (self.df["S4"] == suit4) & (self.df["R4"] == rank4) & (self.df["S5"] == suit5) &
                        (self.df["R5"] == rank5),
                        "Score"
                    ].mean()
                    best_score = max(best_score, score)

        print(best_score)
        if best_score < 1.3:
            return PlayerAction.FOLD, 0
        return PlayerAction.CALL, 0

    @staticmethod
    def get_suit_rank(card_idx):
        card_idx -= 1
        if card_idx // 13 == 1:
            suit = 1
        elif card_idx // 13 == 0:
            suit = 2
        elif card_idx // 13 == 2:
            suit = 3
        else:
            suit = 4

        rank = (card_idx + 2) % 13

        if rank == 0:
            rank = 13

        return suit, rank


if __name__ == "__main__":
    a = StatPlayer("Adam", 1000)
    a.hole_cards = [Card(Rank.TEN, Suit.SPADES), Card(Rank.TEN, Suit.HEARTS)]
    a.action([], [])
