# Yutnori Reinforcement Learning

This project implements a reinforcement-learning environment for **Yutnori**, a traditional Korean board game. It compares rule-based agents, value-based agents, policy-gradient agents, and tactical PPO variants under the same simplified two-player Yutnori rules.

한국어 가이드는 [GUIDE_KO.md](GUIDE_KO.md)를 참고하세요.  
실험 결과 보고서는 [result.md](result.md)를 참고하세요.

## Features

- Two-player Yutnori environment
- Four pieces per player
- Capturing and stacking
- Bonus rolls for `yut` and `mo`
- Bonus roll after capture
- Basic shortcut board graph
- One-hot piece-position observations
- Held yut/mo results that can be consumed in chosen order
- Instant win after 20 consecutive yut/mo bonus rolls
- Action masking for PPO variants
- Value Network, Strategic Value Network, DQN variants, PPO variants, and MCTS + Value agents
- Tournament, fine-tuning, PPO evaluation, and result logging scripts

Back-do is not implemented as a separate move yet. Its probability is folded into `do`.

## Yut Probabilities

| Result | Steps | Probability |
| --- | ---: | ---: |
| do | 1 | 15.3% |
| gae | 2 | 34.6% |
| geol | 3 | 34.6% |
| yut | 4 | 12.0% |
| mo | 5 | 2.6% |

## Project Structure

```text
yut_rl/
  env.py       # Yutnori rules, board movement, environment state
  agents.py    # Baseline, DQN, Value Network, PPO-style agents
  train.py     # Main training and evaluation entry point

agents/
  ppo_agent.py # Masked PPO and capture-aware PPO agents

train/
  train_ppo.py # PPO imitation, curriculum, capture-aware training

experiments/
  tournament.py                # Round-robin tournament and tuning
  improve_agents.py            # Strategic value and DQN improvement experiments
  evaluate_ppo.py              # PPO tournament evaluation and graphs
  strategic_ppo_finetune.py    # StrategicValue-focused PPO fine-tuning

results/
  ppo_eval/
  ppo_training/
  ppo_strategic_finetune/

result.md      # Current experiment report
GUIDE_KO.md    # Korean usage guide
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

## Basic Training

Train the default Value Network agent:

```bash
.venv/bin/python -m yut_rl.train
```

Run a smaller 1,000-episode experiment:

```bash
.venv/bin/python -m yut_rl.train --episodes 1000 --eval-games 200 --eval-interval 200
```

Train Strategic Value Network:

```bash
.venv/bin/python -m yut_rl.train --agent strategic-value --episodes 1000 --eval-games 200
```

Run DQN baselines:

```bash
.venv/bin/python -m yut_rl.train --agent dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent double-dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent dueling-dqn --episodes 1000 --eval-games 200
```

Run policy-gradient baselines:

```bash
.venv/bin/python -m yut_rl.train --agent reinforce --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent a2c --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent ppo --episodes 1000 --eval-games 200
```

## PPO Experiments

Train PPO variants with imitation, curriculum, and capture-aware reward shaping:

```bash
.venv/bin/python train/train_ppo.py --out-dir results/ppo_training --eval-games 1000
```

Train capture-aware PPO variants:

```bash
.venv/bin/python train/train_ppo.py \
  --train-capture-agents \
  --out-dir results/ppo_training \
  --eval-games 1000 \
  --capture-samples 8000 \
  --capture-imitation-epochs 8
```

Evaluate PPO tournament:

```bash
.venv/bin/python experiments/evaluate_ppo.py \
  --games 1000 \
  --model-dir results/ppo_training \
  --out-dir results/ppo_eval
```

Fine-tune PPO specifically against StrategicValue:

```bash
.venv/bin/python experiments/strategic_ppo_finetune.py \
  --analysis-games 1000 \
  --eval-games 1000 \
  --out-dir results/ppo_strategic_finetune
```

## Agents

### Random Agent

Chooses a legal action uniformly at random.

### Rule-based Agent

Uses simple priorities: finish, capture, stacked movement, and distance to goal.

### Strategic Rule-based Agent

Scores legal actions by simulating the move and combining finish progress, captures, shortcut entry, stacking, danger, and opponent counterplay.

### Value Network Agent

Evaluates states instead of actions. For each legal action, it simulates the next state and chooses the action with the highest predicted value.

### Strategic Value Network Agent

Blends Value Network prediction with Strategic Rule-based tactical scoring. This is the strongest non-PPO baseline in the current project.

### PPO Family

The PPO experiments include:

- `ppo_baseline`
- `ppo_masked`
- `ppo_curriculum`
- `ppo_imitation`
- `ppo_capture_imitation`
- `ppo_tactical`
- `ppo_vs_strategic_finetuned`

`ppo_capture_imitation` adds capture-aware reward shaping, tactical oversampling, policy distillation from StrategicValue, and a capture-focused action prior.

## Current Results

Main PPO tournament result:

| Agent | Overall Win Rate | Avg Captures |
| --- | ---: | ---: |
| strategic_value | 72.86% | 2.81 |
| strategic_rule | 72.19% | 2.55 |
| ppo_capture_imitation | 61.76% | 2.77 |
| ppo_tactical | 61.01% | 2.78 |
| ppo_imitation | 41.16% | 1.33 |

StrategicValue direct evaluation:

| Agent | vs StrategicValue Win Rate | Avg Captures |
| --- | ---: | ---: |
| ppo_capture_imitation | 46.0% | 1.84 |
| ppo_vs_strategic_finetuned | 47.6% | 1.88 |

Detailed results are in [result.md](result.md).

## Current Limitations

- Back-do is not implemented as a separate move.
- The environment is a simplified research environment, not a complete official-rule simulator.
- Results depend on seeds, training length, and tournament configuration.
- Some PPO improvements use tactical priors in addition to neural policy learning.
