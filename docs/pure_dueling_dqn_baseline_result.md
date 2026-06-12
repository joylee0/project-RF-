# Pure Dueling Double DQN Baseline 학습 결과

## 실험 목적

project-RF의 초기 방향은 PPO가 아니라 value-based RL인 DQN / Dueling DQN 계열을 사용하여 윷놀이 agent를 학습하는 것이었다.

이 실험은 reward shaping, imitation learning, tactical prior 없이 **raw state + sparse reward + action masking**만 사용했을 때 Dueling Double DQN이 고정 Rule-based Agent를 상대로 어느 정도 학습되는지 확인하기 위해 수행했다.

## 실행 명령어

```bash
source .venv/bin/activate
python train/train_dueling_double_dqn.py --batch-size 512
```

## 실험 설정

| 항목 | 설정 |
| --- | --- |
| Agent | Dueling Double DQN |
| Algorithm | Double Dueling DQN |
| Network | MLP [256, 256] |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Gamma | 0.99 |
| Buffer size | 200,000 |
| Batch size | 512 |
| Learning starts | 10,000 |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay steps | 200,000 |
| Target update | soft update, tau=0.005 |
| State | raw |
| Action | `piece_id * 5 + yut_id` |
| Action masking | true |
| Reward | sparse |
| Reward detail | win +1 / lose -1 / otherwise 0 |
| Reward shaping | false |
| Engineered state | false |
| Imitation learning | false |
| Tactical prior | false |
| Opponent | fixed Rule-based Agent |
| Episodes | 20,000 |
| Seed | 42 |
| Device | CUDA |

## 학습 로그

| Episode | Train Win Rate | Eval Win Rate | Avg Turns |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.190 | 0.204 | 29.2 |
| 2,000 | 0.179 | 0.200 | 28.8 |
| 3,000 | 0.204 | 0.160 | 28.1 |
| 4,000 | 0.196 | 0.202 | 28.6 |
| 5,000 | 0.188 | 0.210 | 29.0 |
| 6,000 | 0.204 | 0.206 | 28.5 |
| 7,000 | 0.213 | 0.228 | 29.5 |
| 8,000 | 0.211 | 0.230 | 28.9 |
| 9,000 | 0.200 | 0.218 | 29.5 |
| 10,000 | 0.220 | 0.230 | 29.3 |
| 11,000 | 0.233 | 0.254 | 29.4 |
| 12,000 | 0.230 | 0.236 | 28.9 |
| 13,000 | 0.253 | 0.254 | 29.2 |
| 14,000 | 0.236 | 0.252 | 28.7 |
| 15,000 | 0.246 | 0.242 | 29.1 |
| 16,000 | 0.237 | 0.264 | 29.6 |
| 17,000 | 0.241 | 0.238 | 29.3 |
| 18,000 | 0.245 | 0.224 | 29.5 |
| 19,000 | 0.257 | 0.248 | 29.5 |
| 20,000 | 0.226 | 0.238 | 29.0 |

## 최종 요약

```json
{
  "agent": "Dueling Double DQN",
  "algorithm": "Double Dueling DQN",
  "setting": "pure_rl_raw_sparse_dueling_double_dqn",
  "state": "raw",
  "reward": "sparse",
  "action_masking": true,
  "episodes": 20000,
  "seed": 42,
  "device": "cuda",
  "batch_size": 512,
  "best_eval_win_rate": 0.264,
  "final_eval_win_rate": 0.238,
  "best_checkpoint": "results/pure_dueling_dqn/dueling_double_dqn_best.pt",
  "latest_checkpoint": "results/pure_dueling_dqn/dueling_double_dqn_latest.pt"
}
```

## 해석

Pure Dueling Double DQN은 학습이 진행되면서 초반 20% 수준에서 최고 26.4%까지 상승했지만, 고정 Rule-based Agent를 안정적으로 이길 정도까지는 도달하지 못했다.

이 결과는 다음을 의미한다.

- raw state와 sparse reward만으로는 윷놀이의 잡기, 완주, 위험 회피 전략을 학습하기 어렵다.
- DQN 계열은 discrete action 환경에 적합하지만, 윷놀이처럼 확률성과 장기 credit assignment가 큰 게임에서는 학습 안정성이 낮을 수 있다.
- project-RF가 이후 PPO 기반 모델, imitation learning, reward shaping, tactical prior를 추가로 실험한 이유를 설명하는 baseline 결과로 사용할 수 있다.

따라서 이 실험은 실패한 실험이라기보다, **pure RL baseline의 한계와 hybrid PPO 설계로 확장한 근거**를 보여주는 중요한 기록이다.
