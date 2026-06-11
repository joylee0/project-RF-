# Design Comparison

## Core Question

The project compares two ways of injecting Yutnori strategy into reinforcement learning:

1. Put tactical knowledge into **policy/reward/prior**.
2. Put tactical knowledge into **observation/state**.

## project-RF

`project-RF` is a hybrid PPO-based implementation.

### Components

- PPO policy network
- 252-dimensional engineered state
- capture-aware dense reward
- StrategicValue teacher imitation
- KL distillation
- inference-time tactical prior
- final logits: `PPO logits + 2.5 * tactical bonus`

### State Features

- both players' piece positions
- pending yut results
- capture possible
- finish possible
- shortcut possible
- danger flag
- progress and distance-to-goal
- engineered / risk-aware encoder experiments

### Reward

| Event | Reward |
| --- | ---: |
| Win | +100 |
| Lose | -100 |
| Finish | +30 |
| Capture | +35 |
| Captured | -35 |
| Missed capture | -10 |
| Danger | -15 |
| Escape danger | +5 |

### Strength

The agent can show tactical behavior with relatively less learning because capture, finish, and danger information are directly encouraged through state, reward, imitation, and prior.

### Limitation

It is not pure PPO. The final behavior is influenced by rule-based teacher knowledge and inference-time tactical score adjustment.

## RL-project

`RL-project` uses sb3-contrib MaskablePPO.

### Components

- Gymnasium environment
- MaskablePPO
- 253-dimensional tactical observation
- terminal reward only
- action selected by PPO network only
- 40M training setup
- common paired evaluation protocol

### Observation Features

- base board state
- stack state
- yut pool
- action-level tactical feature
- legal flag
- capture count
- finish count
- move distance
- `rf_score`

### Reward

| Event | Reward |
| --- | ---: |
| Win | +1 |
| Lose | -1 |
| Otherwise | 0 |

### Strength

The final action is selected by the PPO network without inference-time rule override. Tactical information is provided as observation, so the network can learn how to use it.

### Limitation

Sparse terminal reward generally requires more training. Performance can depend strongly on training budget, seed stability, and opponent distribution.

## Comparison Table

| Dimension | `project-RF` | `RL-project` |
| --- | --- | --- |
| Knowledge location | reward, imitation, policy prior | observation |
| Final action selection | PPO + tactical prior | PPO network only |
| Reward | dense tactical shaping | sparse terminal |
| Sample efficiency | high | requires long training |
| Pure PPO? | no | closer, but with engineered observation |
| Best interpretation | Hybrid RL | PPO + state engineering |

## Interpretation

Both designs are valid RL engineering choices, but they answer different questions.

- `project-RF` asks whether tactical priors and imitation can produce stronger practical behavior.
- `RL-project` asks whether a PPO network can learn strategy when tactical information is made observable.

The common evaluation results show that evaluation context matters. Against the same Rule-based baseline, the agents are nearly tied. In direct match, `project-RF` Hybrid is ahead. In network-only ablation, `RL-project` is stronger.

The main conclusion is that Yutnori RL performance depends not only on the algorithm, but also on state design, reward design, action masking, teacher policies, and inference-time priors.
