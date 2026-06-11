# Evaluation Protocol

## Purpose

The common paired evaluation protocol was introduced to avoid mixing internal tournament results with cross-project comparison. It controls environment rules, random seeds, and first-player advantage.

## Common Environment

The evaluation uses the shared Yutnori rule set:

- 2 players
- 4 pieces per player
- capture and stacking enabled
- automatic shortcut after exact branch landing
- yut/mo bonus turns enabled
- no backdo
- no nak
- no backward movement
- no agent-selected shortcut action
- action space of 20 actions
- illegal actions blocked by action masking

Yut probabilities:

| Result | Steps | Probability |
| --- | ---: | ---: |
| Do | 1 | 0.1536 |
| Gae | 2 | 0.3456 |
| Geol | 3 | 0.3456 |
| Yut | 4 | 0.1296 |
| Mo | 5 | 0.0256 |

## Paired Evaluation

For each base seed, two games are played:

| Game | First Player | Second Player |
| --- | --- | --- |
| A | Agent 1 | Agent 2 |
| B | Agent 2 | Agent 1 |

The standard setting uses:

- 2,500 base seeds
- 2 games per seed
- total 5,000 games
- deterministic policy
- fixed yut probability table
- no future yut-result access
- illegal action count recorded
- evaluation error count recorded

This reduces first-player bias because each seed is evaluated with swapped player order.

## Metrics

The protocol reports:

- total games
- paired seeds
- win rate
- first-player win rate
- second-player win rate
- average turns
- average captures
- average finished pieces
- illegal actions
- evaluation errors
- confidence interval when applicable

## Separation From Internal Results

Internal tournaments and training-stage evaluations are useful for development, but they are not used as final cross-project evidence. Final comparison is based on the common paired protocol.

## Main Result Tables

### Common Rule-based Opponent

| Agent | Win Rate |
| --- | ---: |
| `RL-project` average | 59.76% |
| `project-RF` representative | 59.46% |

### Direct Match

| Metric | Value |
| --- | ---: |
| `project-RF` Hybrid win rate | 53.98% |
| 95% Wilson CI | 53.18% - 54.78% |

### Ablation

| Setting | Win Rate |
| --- | ---: |
| `project-RF` without tactical prior | 17.87% |
| `RL-project` network-only match | 82.13% |

## Reproduction Commands

Validate common env:

```bash
python experiments/validate_common_env.py
```

Run local paired evaluation:

```bash
python experiments/common_paired_evaluation.py \
  --my-agent ppo_capture_imitation \
  --friend-agent friend_ppo \
  --num-paired-seeds 2500 \
  --total-games 5000 \
  --seed 42 \
  --output-dir results/common_paired_eval
```

Run Team PPO direct evaluation:

```bash
python experiments/team_ppo_vs_my_agent_common_eval.py \
  --team-repo /path/to/RL-yutnori \
  --model-path /path/to/model.zip \
  --training-seed 1 \
  --num-paired-seeds 2500 \
  --seed-start 100000 \
  --output-dir results/team_ppo_vs_my_agent_common_eval/seed1
```
