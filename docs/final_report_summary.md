# Final Report Summary

## Project Theme

This project studies reinforcement learning in Yutnori, a Korean traditional board game. The final report compares two design philosophies:

1. **`project-RF`**: Hybrid PPO with StrategicValue teacher imitation and inference-time tactical prior.
2. **`RL-project`**: MaskablePPO with tactical observation and network-only action selection.

The shared goal is to evaluate whether RL agents can learn useful Yutnori strategies under a common environment and a common paired evaluation protocol.

## RL Problem Definition

- **State**: board state, piece positions, yut results, and optional tactical features.
- **Action**: select one piece and one currently available yut result.
- **Reward**: win/loss terminal reward or shaped tactical reward.
- **Policy**: maps state to action probabilities.
- **Environment**: implements Yutnori transition dynamics, capture, stacking, shortcut, and terminal condition.

## Design Summary

| Item | `project-RF` | `RL-project` |
| --- | --- | --- |
| Main model | Hybrid PPO | MaskablePPO |
| Knowledge location | policy, reward, imitation, tactical prior | observation features |
| State | 252-dimensional engineered state | 253-dimensional tactical observation |
| Reward | dense capture-aware reward | terminal reward only |
| Final action | PPO logits + tactical prior | PPO network only |
| Classification | Hybrid RL | PPO with state engineering |

## Main Results

### Common Rule-based Opponent

| Agent | Win Rate |
| --- | ---: |
| `RL-project` average | 59.76% |
| `project-RF` representative | 59.46% |

The two agents perform almost equally against the common Rule-based baseline.

### Direct Paired Evaluation

| Metric | Value |
| --- | ---: |
| `project-RF` Hybrid win rate | 53.98% |
| 95% Wilson CI | 53.18% - 54.78% |

In direct paired evaluation, `project-RF` Hybrid is ahead.

### Tactical Prior Ablation

| Setting | Win Rate |
| --- | ---: |
| `project-RF` without tactical prior | 17.87% |
| `RL-project` network-only match | 82.13% |

The result shows that the tactical prior is a major reason for `project-RF`'s direct-match advantage.

## Key Interpretation

- A single win rate is not enough to judge agent quality.
- Against the same baseline, the two agents are nearly tied.
- In direct match, the hybrid policy/prior design has an advantage.
- Without the inference-time prior, `RL-project`'s PPO network is stronger.
- The project mainly shows how state, reward, imitation, and prior design affect RL behavior.

## Limitations

- Training budgets are not identical.
- `project-RF` includes hybrid components, so it should not be described as pure PPO.
- `RL-project` uses long training and multiple seeds, while some `project-RF` runs are representative checkpoint based.
- The main comparison uses the legacy no-backdo environment.

## Conclusion

The project demonstrates that Yutnori RL performance depends heavily on where domain knowledge is injected. `project-RF` achieves strong tactical behavior through imitation and policy prior, while `RL-project` keeps action selection network-only and places tactical information in the observation.
