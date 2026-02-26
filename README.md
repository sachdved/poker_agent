# LLM Poker Player

Fine-tuning a Large Language Model to play poker through self-interaction using reinforcement learning.

## Overview

This project explores training language models to excel at strategic decision-making by having them play poker against themselves. Unlike traditional AI approaches that use handcrafted strategies, this method uses the LLM's natural language understanding capabilities to learn poker strategies from experience.

## Architecture

### Core Components

- **LLM Feature Extractor**: Uses a Qwen3 (1.7B parameters) model as a feature extraction layer. The LLM processes game states and provides contextual embeddings of the poker situation.

- **Policy Network**: A feedforward neural network that takes the LLM embeddings along with game state features (cards, positions, stack sizes) and predicts action probabilities.

- **Poker Sequence Embedder**: Encodes game events (streets, table positions, actions) into embeddings for the LLM input.

- **Self-Play Simulation**: Complete poker game simulation with multiple players, handling all game rules including betting rounds, pot management, and hand evaluation.

### Model Components

```python
- Cards Embedder: Encodes card ranks and suits
- Street Positional Encoding: Encodes which betting round
- Table Position Encoding: Encodes player's position at the table
- Action Encoding: Encodes previous actions
- Pot Size Sequence: Tracks accumulated pot across rounds
- Policy Model: MLP that outputs action probabilities (fold, check/call, bet sizes, raise sizes)
```

## Training Approach

### Self-Play Reinforcement Learning

The model learns through self-play using a GRPO framework:

1. **Rollout Phase**: Multiple agents play complete poker hands against each other (starting from the same deck configuration).
2. **Reward Calculation**: Based on final chip stack at hand end
3. **Advantage Estimation**: Within each group of rollouts sharing the same deck configuration, advantages are normalized relative to the group mean reward, following the GRPO framework and eliminating the need for a separate value network
4. **Policy Update**: Update policy using gradient ascent on advantage-weighted log probabilities

### Key Features

- **Multiple Players**: Supports N-player games (tested with 2-8 players)
- **Full Game Logic**: Preflop, flop, turn, river with proper betting structures
- **Gradient Backpropagation**: Only policy network gradients flow back (LLM frozen)
- **Batch Processing**: Parallel simulation across multiple hands
- **Entropy Regularization**: Encourages exploration during training

## File Structure

```
├── agent.py                    # PokerAgent class - combines LLM and policy model
├── llm_modules.py             # LLM loading and hook utilities
├── ml_modules.py              # Policy model and card embeddings
├── ml_ops_utils.py            # Training utilities and GPU management
├── simulation.py              # Poker hand simulation logic
├── sequence_modules.py        # Game state embedding modules
├── utils.py                   # Helper functions for poker game
├── hand_strength.py           # Hand evaluation and winner determination
└── 2026-02-22_generalize_to_n_players.ipynb  # Training notebook
```

## Usage

### Basic Setup

```python
from agent import PokerAgent
from ml_modules import PolicyModel
from llm_modules import load_model
from sequence_modules import *
from utils import *
from simulation import simulate_hand

# Load LLM
tokenizer, model = load_model("./models/qwen3-1point7b/")

# Build components
cards = Cards(device="cuda")
street_embedder = StreetPositionalEncoding(...)
table_position_embedder = TablePositionalEncoding(...)
action_embedder = ActionEncoding(...)
pot_size_embedder = PotSizeSequenceEmbedder(...)
poker_sequence_embedder = PokerSequenceEmbedder(...)
policy_model = PolicyModel(...)

# Create agent
agent = PokerAgent(
    cards, street_embedder, table_position_embedder,
    action_embedder, pot_size_embedder, poker_sequence_embedder,
    model, policy_model, device="cuda", llm_train=False
)
```

### Training

The training notebook demonstrates the complete self-play training loop:

```python
# Build multiple agents for self-play
agents = [build_agent(...) for _ in range(num_players)]
optimizers = [torch.optim.AdamW(agent.parameters(), lr=1e-4) for agent in agents]

# Run training loop
for step in range(num_steps):
    # Run multiple rollouts
    for rollout in range(num_rollouts):
        # Simulate complete hand
        table, ..., seated_agents = build_hand(...)
        street_idxs, ..., table = simulate_hand(...)

        # Compute rewards and advantages
        rewards = determine_winner(...)
        advantages = compute_advantages(rewards)

        # Update policy networks
        for player_idx in range(num_players):
            loss = compute_advantage_loss(...)
            loss.backward()
            optimizers[player_idx].step()
```

## Technical Details

### Action Space

The policy model outputs probabilities over 21 possible actions:
- Action 0: Post small blind
- Action 1: Post big blind
- Action 2: Fold
- Action 3: Check
- Action 4: Call
- Actions 5-16: Bet/raise sizes (varying bet sizes)
- Action 17-20: End of hand/raise cap actions

### Neural Network Architecture

- **LLM Layer**: Qwen3 (28 layers, 2048 hidden dimensions)
- **Hook Point**: Uses layer 27's post-attention layer for activation extraction. Other layers to be explored.
- **Policy Network**: Multi-layer MLP with [1024, 2048, 512, 256] hidden dimensions

### Training Hyperparameters

- **Learning Rate**: 1e-4 (AdamW optimizer)
- **Batch Size**: 1024 parallel hands
- **Entropy Coefficient**: 0.01-0.1 for exploration
- **Optimization**: Only policy network parameters are updated

## Key Innovation

**Why GRPO over CFR?**

This research explores whether we can learn complex strategic behaviors without explicitly constructing game trees. Traditional approaches like Counterfactual Regret Minimization (CFR) require exploring the full game tree to compute equilibrium strategies, which becomes intractable as game complexity grows.

The central hypothesis of this project is that **pre-trained language models already contain rich semantic and strategic understanding** that can be harnessed through RL without explicit game tree enumeration:

1. **Tree Sampling vs. Tree Enumeration**: Instead of exhaustively searching the game tree, we use the LLM's generative capabilities to sample action sequences, letting the model explore relevant branches based on its pre-trained knowledge

2. **Semantic Knowledge Integration**: The LLM has learned patterns from human language that encode strategic reasoning, probability estimation, and contextual understanding - these can be extracted and refined through RL

3. **Scalability to Complex Games**: This approach scales better than tree-based methods, as the LLM can generalize to novel situations without needing exhaustive training on all possible game states

4. **Self-Play without Equilibrium**: Unlike CFR which converges to Nash equilibrium, this approach allows the model to learn through self-interaction and competition, potentially more rapidly learning exploitative strategies against non-equilibrium opponents.

   
## Preliminary Results

Early training runs show the agent adapts to trivial cases 
(e.g., folding clearly losing hands) within 20-30 hands, 
suggesting the LLM embeddings provide a useful prior for 
strategic reasoning. Evaluation against Nash equilibrium 
baselines is ongoing.

## Potential Applications

Beyond poker, this methodology could be applied to:
- Game-playing AI development
- Strategic decision-making in complex systems
- Fine-tuning LLMs for specific domains through self-play
- Reinforcement learning with LLMs

## License

[Add your license here]

## Acknowledgments

- Qwen model family for the LLM implementation
- Research on self-play reinforcement learning for game AI