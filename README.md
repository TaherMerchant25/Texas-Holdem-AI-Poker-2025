# ♠️ Texas Hold'em Poker AI Engine ♣️

A modular Python engine for simulating and experimenting with Texas Hold'em poker, featuring AI and rule-based players, hand evaluation, and a customizable game loop.

---

## ✨ Features

- 🃏 **Texas Hold'em Poker Simulation**: Complete game logic including betting rounds, blinds, and showdown.
- 🤖 **AI & Rule-Based Players**: Adaptive AI (e.g., `AdaptiveRaisePlayer`), random bots, and user input players.
- 🏆 **Hand Evaluation**: Robust hand evaluator for ranking poker hands (high card to royal flush).
- 🧩 **Extensible Design**: Easily add new player strategies or modify game rules.
- 💻 **Command-Line Interface**: Play against bots or test AI strategies from the terminal.

---

## 🚀 Getting Started

### 🛠️ Requirements

- Python 3.8+
- No external dependencies required

### 📦 Installation

Clone this repository and run locally:
```
git clone https://github.com/yourusername/poker-ai-engine.git
cd poker-ai-engine
```

---

## 🗂️ File Structure

| File/Folder         | Description                                    |
|---------------------|------------------------------------------------|
| `main.py`           | Entry point; runs a sample game loop           |
| `game.py`           | Core game logic and state management           |
| `card.py`           | Card, Deck, Suit, and Rank definitions         |
| `hand_evaluator.py` | Poker hand evaluation logic                    |
| `baseplayers.py`    | Rule-based and random player bots              |
| `Raisep.py`         | AdaptiveRaisePlayer AI implementation          |
| `player.py`         | Player base class and enums                    |

---

## 🕹️ Usage

### ▶️ Run a Sample Game


- The game will run several hands between different AI players.
- Modify `main.py` to customize player types or stack sizes.

### 👥 Adding Players

Players are defined in their respective modules and added to the game in `main.py`:

```
from Raisep import AdaptiveRaisePlayer
from baseplayers import RaisePlayer

players = [
AdaptiveRaisePlayer("AI", 1000),
RaisePlayer("Bob", 1000),
# Add more players as needed
]
```

---

## 🏗️ Key Components

### 🔄 Game Flow

- **Phases**: Setup → Pre-Flop → Flop → Turn → River → Showdown
- **Actions**: Fold, Call, Check, Bet, Raise, All-In
- **Blinds**: Big blind enforced, no small blind by default
- **Showdown**: Hand evaluator determines the winner(s) and splits the pot

### 🃏 Hand Evaluation

- Evaluates all possible 5-card combinations from 7 cards (hole + community).
- Supports all standard poker hands: Pair, Two Pair, Straight, Flush, Full House, Four of a Kind, Straight Flush, Royal Flush.

### 🧑‍💻 Player Types

| Player Class            | Description                                   |
|-------------------------|-----------------------------------------------|
| `AdaptiveRaisePlayer`   | Adjusts aggression based on opponents         |
| `RandBot`/`AggresiveRandBot` | Randomized betting and folding         |
| `RaisePlayer`           | Always raises if possible                     |
| `InputPlayer`           | Human input via console                       |

---

## 🤖 Example: AdaptiveRaisePlayer

- Dynamically models opponents' aggression and fold frequency.
- Estimates hand strength using the hand evaluator.
- Adjusts raise size and bluffing based on game context and opponent tendencies.

---

## 🛠️ Customization

- **Add new player strategies**: Inherit from `Player` and implement the `action` method.
- **Change game rules**: Modify `game.py` for custom betting, blinds, or phase logic.
- **Integrate with GUI or Web**: The modular design allows easy extension.

---

## 📄 License

MIT License (or specify your project's license)

---

## 🙏 Credits

Developed by [Your Name or Team].  
Special thanks to contributors and open-source libraries.

---

## 🖥️ Example Output

```
Phase: pre-flop
Pot: 40
Community cards: []
Players:
→ AI: $980 [hidden] active
Bob: $980 [hidden] active
...
AI's turn
Your cards: ['Q♠', 'J♦']
Available actions:

Check

Bet
```

---

## 📬 Contact

For questions or contributions, open an issue or pull request on GitHub.

---

**Enjoy experimenting with poker AI! 🃏🤖**
