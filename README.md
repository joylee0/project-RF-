# Yutnori Reinforcement Learning

This project implements a reinforcement-learning playground for **Yutnori**, a traditional Korean board game. It includes a simplified two-player Yutnori environment and several agents for baseline comparison.

The current main model is a **Value Network agent** that selects actions by simulating legal moves and comparing the value of the resulting next states. A DQN baseline is also included for comparison.

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
- Random, rule-based, DQN, and Value Network agents
- Training, evaluation, checkpoint saving, and continued training

## Yut Probabilities

The environment currently uses the standard combinational probabilities without back-do:

| Result | Steps | Probability |
| --- | ---: | ---: |
| do | 1 | 4/16 |
| gae | 2 | 6/16 |
| geol | 3 | 4/16 |
| yut | 4 | 1/16 |
| mo | 5 | 1/16 |

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

The default command trains the Value Network agent for 50,000 episodes.

```bash
.venv/bin/python -m yut_rl.train
```

Run a smaller 1,000-episode experiment:

```bash
.venv/bin/python -m yut_rl.train --episodes 1000 --eval-games 200 --eval-interval 200
```

Useful options:

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --eval-games 200 \
  --eval-interval 5000 \
  --lr 0.001 \
  --gamma 0.97 \
  --batch-size 64 \
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

### DQN Agent

Learns action-values `Q(s, a)` and directly outputs values for the four piece choices.

### Value Network Agent

Evaluates states rather than actions. For each legal action, it simulates the next state and chooses the move with the highest predicted state value. This is closer to the approach used in previous Yutnori reinforcement-learning research than the basic DQN baseline.

## Current Limitations

- Back-do is not implemented yet.
- The state representation is still compact and can be improved.
- The current Value Network is inspired by prior work but is not a full reproduction of the paper.
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

