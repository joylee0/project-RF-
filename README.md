# Yutnori Reinforcement Learning

This project implements a reinforcement-learning playground for **Yutnori**, a traditional Korean board game. It includes a simplified two-player Yutnori environment and several agents for baseline comparison.

The default model is a **Value Network agent**, closer to the approach used in prior Yutnori reinforcement-learning work. Tabular Q-learning and DQN baselines are also included for comparison.

한국어 사용 가이드는 [GUIDE_KO.md](GUIDE_KO.md)를 참고하세요.

## Features

- Two-player Yutnori environment
- Four pieces per player
- Capturing
- Stacking
- Bonus rolls for `yut` and `mo`
- Bonus roll after capture
- Basic shortcut board graph
- No back-do yet
- One-hot piece-position observations
- Held yut/mo roll results that can be consumed in chosen order
- Consecutive yut/mo bonus rolls capped at four
- Expanded actions: roll result choice plus piece choice
- Terminal loss reward of `-1`
- DQN and Value Network hidden size default of 256
- Multi-opponent self-play for the Value Network agent
- Random, rule-based, Tabular Q-learning, DQN, and Value Network agents
- Training, evaluation, checkpoint saving, and continued training

## Yut Probabilities

The environment uses the probability distribution from the referenced Yutnori reinforcement-learning paper. Back-do is not implemented yet, so its 3.8% probability is folded into do. The rounded do/gae/geol/yut/mo probabilities are normalized internally to sum to 1.

| Result | Steps | Probability |
| --- | ---: | ---: |
| do | 1 | 15.3%, including back-do |
| gae | 2 | 34.6% |
| geol | 3 | 34.6% |
| yut | 4 | 12.0% |
| mo | 5 | 2.6% |

## Project Structure

```text
yut_rl/
  env.py       # Yutnori environment and board movement rules
  agents.py    # Random, rule-based, DQN, and Value Network agents
  train.py     # Training and evaluation entry point

GUIDE_KO.md    # Korean guide
README.md      # Project overview
requirements.txt
```

## Installation

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Quick Test

```bash
.venv/bin/python -m yut_rl.train --episodes 20 --eval-games 20 --eval-interval 0
```

## Train the Value Network Agent

The default command trains the Value Network agent.

```bash
.venv/bin/python -m yut_rl.train
```

Run a smaller 100-episode experiment:

```bash
.venv/bin/python -m yut_rl.train --episodes 100 --eval-games 50 --eval-interval 50
```

## Longer Value Network Training

This command trains the Value Network agent for 50,000 episodes.

```bash
.venv/bin/python -m yut_rl.train --agent value --episodes 50000
```

Run a smaller 1,000-episode experiment:

```bash
.venv/bin/python -m yut_rl.train --episodes 1000 --eval-games 200 --eval-interval 200
```

Configure the self-play opponent pool:

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --snapshot-interval 1000 \
  --opponent-pool-size 5 \
  --rule-opponent-weight 4.0
```

Useful options:

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --eval-games 200 \
  --eval-interval 5000 \
  --gamma 0.97 \
  --hidden-dim 256 \
  --epsilon-start 0.8 \
  --epsilon-end 0.05 \
  --save-model checkpoints/value.pt
```

## Run the DQN Baseline

```bash
.venv/bin/python -m yut_rl.train --agent dqn --episodes 1000 --eval-games 200 --eval-interval 200
```

## Continue Training

```bash
.venv/bin/python -m yut_rl.train \
  --episodes 20000 \
  --load-model checkpoints/value.pt \
  --save-model checkpoints/value.pt
```

## Agents

### Random Agent

Chooses one legal piece uniformly at random.

### Rule-based Agent

Uses simple hand-written priorities such as finishing pieces, capturing opponents, moving stacked pieces, and reducing distance to finish.

### Tabular Q-learning Agent

Learns Q-values in a Python dictionary without PyTorch. This is the default first-run agent because it works with only the Python standard library.

### DQN Agent

Learns action-values `Q(s, a)` over the expanded action space: five possible roll results times four pieces.

### Value Network Agent

Evaluates states rather than actions. For each legal action, it simulates the next state and chooses the move with the highest predicted state value. This is closer to the approach used in previous Yutnori reinforcement-learning research than the basic DQN baseline.

During training, it can play against the current model, random/rule-based baselines, and frozen snapshots of older versions of itself.

## Current Limitations

- Back-do is not implemented yet.
- The state representation now uses one-hot piece positions, but it is still smaller than a full paper reproduction.
- The current Value Network is inspired by prior work but is not a full reproduction of the paper.
- Checkpoints created before the one-hot/action-space change may not load into the current model.
- Evaluation results depend on random seeds and training length.

## GitHub Upload

Create an empty GitHub repository first, then run:

```bash
git init
git add .
git commit -m "Initial yutnori reinforcement learning project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The `.gitignore` file excludes virtual environments, Python cache files, and model checkpoints.
