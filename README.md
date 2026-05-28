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
- Instant win after 20 consecutive yut/mo bonus rolls
- Expanded actions: roll result choice plus piece choice
- Terminal loss reward of `-1`
- Hybrid reward shaping by default: terminal win/loss plus scaled dense rewards
- DQN and Value Network hidden size default of 256
- DQN-style agents train on the state after the opponent response when the turn changes
- Multi-opponent self-play for the Value Network agent
- Random, rule-based, strategic rule-based, Tabular Q-learning, DQN, Double DQN, Dueling DQN, REINFORCE, A2C, PPO, Value Network, strategic Value Network, and MCTS + Value Network agents
- 2-step lookahead for the Value Network agent by default
- Gradient clipping, PPO minibatch-style rollout collection, and normalized policy-gradient returns
- Best checkpoint saving by Rule-based win rate
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
  --rule-opponent-weight 4.0 \
  --lookahead-depth 2
```

For a stronger Rule-based opponent focus, increase the Rule-based opponent weight and keep best checkpoint saving on:

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --eval-games 300 \
  --eval-interval 2500 \
  --rule-opponent-weight 6.0 \
  --random-opponent-weight 0.25 \
  --snapshot-opponent-weight 1.5 \
  --lookahead-depth 2 \
  --best-model checkpoints/best_value_vs_rule.pt
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

## Additional Agents

```bash
.venv/bin/python -m yut_rl.train --agent double-dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent dueling-dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent strategic --eval-games 200
.venv/bin/python -m yut_rl.train --agent strategic-value --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent reinforce --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent a2c --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent ppo --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent mcts-value --episodes 1000 --eval-games 200
```

## Performance Tuning

Good starting commands for each neural family:

```bash
# More stable Q-learning baseline
.venv/bin/python -m yut_rl.train --agent double-dqn --episodes 20000 --target-sync 100

# Stronger DQN variant for action-value learning
.venv/bin/python -m yut_rl.train --agent dueling-dqn --episodes 20000 --target-sync 100

# Best paper-like baseline in this project
.venv/bin/python -m yut_rl.train --agent value --episodes 50000 --lookahead-depth 2 --rule-opponent-weight 6.0

# Value Network with tactical Yutnori heuristics blended into action selection
.venv/bin/python -m yut_rl.train --agent strategic-value --episodes 50000 --lookahead-depth 2 --heuristic-weight 0.35

# Policy-gradient baseline with multiple rollouts per update
.venv/bin/python -m yut_rl.train --agent ppo --episodes 20000 --rollout-episodes 8 --ppo-epochs 4

# Search at evaluation time, slower but often stronger after value training
.venv/bin/python -m yut_rl.train --agent mcts-value --episodes 50000 --mcts-simulations 64 --mcts-rollout-depth 8
```

Useful training knobs:

- `--reward-mode hybrid`: default. Uses terminal `+1/-1` with scaled capture/finish/step rewards.
- `--dense-reward-scale`: lowers or raises intermediate reward strength in hybrid mode.
- `--grad-clip`: prevents unstable neural updates.
- `--heuristic-weight`: controls how strongly strategic Value Network follows tactical heuristics.
- `--target-sync`: controls target-network update frequency for DQN and Value Network.
- `--rollout-episodes`: collects more policy-gradient games before each update.
- `--best-model`: saves the checkpoint with the best Rule-based win rate.

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

### Strategic Rule-based Agent

Scores each legal move by simulating it and combining finish progress, captures, stack movement, shortcut entry, and exposure to opponent capture threats.

The current default also includes a counterplay penalty. It evaluates the opponent's best immediate reply after the candidate move, which pushed the strategic rule-based agent above 55% in the 1,000-game comparison.

### Tabular Q-learning Agent

Learns Q-values in a Python dictionary without PyTorch. This is the default first-run agent because it works with only the Python standard library.

### DQN Agent

Learns action-values `Q(s, a)` over the expanded action space: five possible roll results times four pieces.

It is a direct value-based baseline: given the current observation, it predicts how good each legal action is. In this project, DQN transitions wait until the trained player gets control again when the turn passes to the opponent, so the next state is represented from the trained player's perspective.

### Double DQN Agent

Separates next-action selection from target-network evaluation to reduce Q-value overestimation.

This is generally a stronger comparison point than plain DQN for this game because Yutnori has long tactical chains caused by captures, stacking, and bonus rolls.

### Dueling DQN Agent

Uses separate value and advantage heads before combining them into action-values.

The implementation combines the dueling architecture with Double DQN-style target calculation, so it is the strongest DQN-family baseline in this codebase.

### REINFORCE Agent

Learns a policy directly from complete game returns. It is simple and useful as a policy-gradient baseline, but it can be noisy because one win or loss affects many earlier actions.

### A2C Agent

Uses an actor-critic structure: the actor chooses actions, while the critic estimates state value. This usually gives a more stable update signal than plain REINFORCE.

### PPO Agent

Uses clipped policy updates to keep the policy from changing too aggressively. The training loop collects multiple rollout games per update and normalizes returns/advantages by default.

### Value Network Agent

Evaluates states rather than actions. For each legal action, it simulates the next state and chooses the move with the highest predicted state value. This is closer to the approach used in previous Yutnori reinforcement-learning research than the basic DQN baseline.

During training, it can play against the current model, random/rule-based baselines, and frozen snapshots of older versions of itself.

### Strategic Value Network Agent

Uses the same Value Network, but blends the learned state-value score with the strategic heuristic score during action selection. This keeps early or weak checkpoints from ignoring tactical Yutnori decisions such as capture risk and shortcut timing.

### MCTS + Value Network Agent

Uses the Value Network as a light search evaluator for legal actions.

For each legal move, it runs several short rollouts and then uses the Value Network to evaluate the resulting state. More simulations can improve decisions but make evaluation slower.

## Compare Improved Agents

```bash
.venv/bin/python -m yut_rl.compare --games 1000
.venv/bin/python -m yut_rl.compare --games 1000 --value-model checkpoints/value.pt
```

The comparison reports win rate, average turns, captures, finished pieces, and average reward while alternating first-player order.

To search for high-win-rate parameter combinations:

```bash
.venv/bin/python -m yut_rl.tune --games 100 --confirm-games 1000 --top-k 3 --value-model checkpoints/value.pt
```

Current 1,000-game result:

- `strategic_rule_based`: 57.4%
- `strategic_value`: 72.5%

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
