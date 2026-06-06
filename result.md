# 윷놀이 강화학습 프로젝트 결과 보고서

## 1. 프로젝트 개요

이 프로젝트는 한국 전통 놀이인 윷놀이를 강화학습 환경으로 구현하고, 여러 에이전트의 전략 성능을 비교하기 위한 실험 프로젝트이다. 기본 환경은 2인 대전, 플레이어당 말 4개, 잡기, 업기, 지름길, 윷/모 추가 던지기, 잡기 후 추가 던지기 규칙을 포함한다.

초기 목표는 DQN, Value Network, PPO 같은 강화학습 모델을 같은 윷놀이 환경에서 비교하는 것이었고, 이후 실험은 `strategic_value`를 강한 기준선으로 두고 PPO 계열을 개선하는 방향으로 진행되었다.

## 2. Environment

환경 파일은 `yut_rl/env.py`이다.

### 기본 규칙

- 플레이어 수: 2명
- 말 개수: 각 플레이어 4개
- 지원 규칙:
  - 잡기
  - 업기
  - 잡았을 때 추가 던지기
  - 윷/모 추가 던지기
  - 지름길
  - 윷/모 연속 20회 시 즉시 승리
- 미지원 규칙:
  - 뒷도는 아직 별도 행동으로 구현하지 않음

### 윷 확률

논문에서 사용한 확률을 기준으로 하되, 뒷도는 아직 구현하지 않았기 때문에 도 확률에 합산했다.

| 결과 | 이동 칸 수 | 확률 |
| --- | ---: | ---: |
| 도 | 1 | 15.3% |
| 개 | 2 | 34.6% |
| 걸 | 3 | 34.6% |
| 윷 | 4 | 12.0% |
| 모 | 5 | 2.6% |

윷/모가 나오면 즉시 행동하지 않고, 윷/모가 더 이상 나오지 않을 때까지 결과를 모아 둔 뒤 보유한 결과 중 하나를 선택해서 사용한다.

## 3. State Representation

기본 `YutEnv.observe()`는 말 위치와 현재 보유한 윷 결과 중심의 one-hot 상태를 제공한다.

PPO 개선 과정에서는 별도 state encoder를 `agents/ppo_agent.py`에 추가했다. PPO용 state는 기존 238차원에서 252차원으로 확장되었다.

포함 feature:

- 내 말 4개의 위치 one-hot
- 상대 말 4개의 위치 one-hot
- 현재 보유한 윷 결과
- 내 완주 말 수
- 상대 완주 말 수
- 잡을 수 있는 말 존재 여부
- 완주 가능 여부
- 지름길 진입 가능 여부
- 잡힐 위험 여부
- 각 말의 goal까지 남은 거리

이 state는 PPO가 “잡기 가능”, “위험 회피”, “완주 우선” 같은 전략 feature를 더 직접적으로 학습할 수 있도록 설계했다.

## 4. Reward 설계

초기 환경 보상은 작은 중간 보상과 최종 승패 보상을 함께 사용하는 hybrid 형태였다.

PPO capture-aware 실험에서는 보상 shaping을 별도로 강화했다.

| 항목 | 보상 |
| --- | ---: |
| 승리 | +100 |
| 패배 | -100 |
| 내 말 완주 | +30 |
| 상대 말 잡기 | +35 |
| 내 말 잡힘 | -35 |
| 잡을 수 있는데 잡지 않음 | -10 |
| 지름길 진입 | +8 |
| 전진 거리 | 거리 차이 x 0.5 |
| 위험 위치 이동 | -15 |
| 위험에서 안전 위치로 이동 | +5 |
| 턴 증가 패널티 | -0.1 |

이후 `strategic_value` 전용 fine-tuning에서는 패배 로그 분석 결과를 반영해 위험 위치 이동과 상대 counterplay를 더 강하게 반영했다.

## 5. Agents

### Baseline Agents

- `RandomAgent`: 가능한 행동 중 무작위 선택
- `RuleBasedAgent`: 완주, 잡기, 진행도를 기준으로 단순 점수화
- `StrategicRuleBasedAgent`: 잡기, 완주, 진행도, 위험도, 지름길, 상대 반격 가능성을 함께 계산

### Value-based Agents

- `TabularQAgent`
- `DQNAgent`
- `DoubleDQNAgent`
- `DuelingDQNAgent`
- `ValueNetworkAgent`
- `StrategicValueNetworkAgent`
- `MCTSValueAgent`

`StrategicValueNetworkAgent`는 Value Network의 상태 가치 예측과 Strategic Rule-based의 전략 점수를 섞어 행동을 고른다. 현재 전체 실험에서 가장 강한 기준선 역할을 한다.

### Policy-gradient Agents

- `ReinforceAgent`
- `A2CAgent`
- `PPOAgent`
- `MaskedPPOAgent`
- `CaptureAwarePPOAgent`

PPO 개선 과정:

1. 기본 PPO
2. action masking 적용 PPO
3. curriculum PPO
4. StrategicValue teacher imitation PPO
5. capture-aware PPO
6. strategic_value 전용 fine-tuned PPO

## 6. PPO 개선 과정

### Action Masking

PPO policy logits에서 현재 legal action이 아닌 행동은 `-1e9`로 masking했다. 이를 통해 PPO가 현재 사용할 수 없는 말/윷 결과 조합을 선택하지 않도록 했다.

### Imitation Pretraining

StrategicValue agent를 teacher로 사용해 state-action pair를 수집했다. 이후 단순 cross entropy뿐 아니라 teacher의 action score를 softmax 분포로 바꾸고, PPO policy와 KL divergence loss로 distillation하는 방식을 추가했다.

중요 state는 oversampling했다.

- capture possible state: 5배
- danger state: 5배
- finish possible state: 3배
- shortcut possible state: 2배

### Capture-aware PPO

`ppo_capture_imitation`은 capture reward를 강화하고, inference 단계에서 잡기/완주/위험 회피를 반영하는 tactical prior를 추가한 PPO agent이다.

### StrategicValue 전용 Fine-tuning

`ppo_vs_strategic_finetuned`는 `strategic_value` 직접 대결 승률을 높이기 위해 추가한 모델이다.

fine-tuning 상대 구성:

- StrategicValueNetworkAgent: 80%
- StrategicRuleBasedAgent: 10%
- PPO self-play: 10%

패배 replay를 분석해 critical state를 oversampling하고, teacher distillation을 추가로 수행했다.

## 7. 실험 결과

### PPO Tournament 결과

결과 파일: `results/ppo_eval/ppo_tournament_summary.csv`

| Agent | Overall Win Rate | Avg Captures | Capture Success Rate | Avg Finished Pieces |
| --- | ---: | ---: | ---: | ---: |
| strategic_value | 72.86% | 2.81 | 89.36% | 3.55 |
| strategic_rule | 72.19% | 2.55 | 84.85% | 3.55 |
| ppo_capture_imitation | 61.76% | 2.77 | 91.14% | 3.34 |
| ppo_tactical | 61.01% | 2.78 | 90.66% | 3.33 |
| ppo_imitation | 41.16% | 1.33 | 38.88% | 2.98 |
| ppo_baseline | 32.23% | 1.32 | 41.54% | 2.38 |
| ppo_curriculum | 30.53% | 1.34 | 40.08% | 2.37 |
| ppo_masked | 28.27% | 1.33 | 39.30% | 2.06 |

주요 해석:

- 기본 PPO 계열은 strategic 계열보다 낮은 성능을 보였다.
- `ppo_imitation`은 PPO 중 가장 좋은 초기 개선 모델이었지만, 잡기 횟수가 부족했다.
- capture-aware reward와 tactical prior를 추가한 뒤 `ppo_capture_imitation`은 overall win rate 61.76%까지 상승했다.
- 잡기 성공률은 91.14%로 가장 높았다.

### StrategicValue 직접 대결

결과 파일: `results/ppo_strategic_finetune/direct_vs_strategic_value.csv`

| Agent | vs StrategicValue Win Rate | Avg Captures | Avg Finished Pieces |
| --- | ---: | ---: | ---: |
| ppo_capture_imitation | 46.0% | 1.84 | 3.09 |
| ppo_vs_strategic_finetuned | 47.6% | 1.88 | 3.12 |

별도 tactical weight search에서는 `ppo_vs_strategic_finetuned`가 `strategic_value` 상대로 최대 49.7%까지 도달했다.

### 패배 로그 분석

결과 파일: `results/ppo_strategic_finetune/strategic_loss_summary.csv`

| 패배 원인 | 횟수 |
| --- | ---: |
| 잡을 수 있었는데 잡지 않음 | 85 |
| 완주 가능했는데 완주하지 않음 | 7 |
| 잡힐 위험 위치로 이동 | 2761 |
| 지름길 진입 기회 놓침 | 373 |
| 후반부 말 선택 실수 | 85 |

가장 큰 약점은 “잡기 부족”보다 “상대에게 잡힐 위험 위치로 이동”이었다. 이후 tactical prior에 상대 counterplay penalty와 endgame 우선순위를 강화했다.

## 8. 최종 결론

현재 가장 강한 기준선은 `strategic_value`이다. 다만 PPO 계열도 capture-aware imitation과 tactical prior를 추가하면서 전체 tournament 승률 60% 이상을 달성했다.

최종 주요 성과:

- PPO 전체 승률: `ppo_imitation` 41.16% -> `ppo_capture_imitation` 61.76%
- PPO 평균 잡기: 1.33 -> 2.77
- StrategicValue 직접 대결: 약 37.3% 수준 -> 47.6%
- Tactical weight search 최고값: 49.7%

현재 PPO는 `strategic_value`와 거의 대등한 직접 승률에 접근했지만, seed와 평가 방식에 따라 편차가 있다. 이후 개선 방향은 더 긴 strategic_value self-play, 위험 회피 value head, opponent model 기반 counterplay 예측, 뒷도 규칙 추가가 될 수 있다.

## 9. 주요 결과 파일

- `results/ppo_eval/ppo_tournament_summary.csv`
- `results/ppo_eval/ppo_tournament_matchups.csv`
- `results/ppo_eval/best_ppo.json`
- `results/ppo_strategic_finetune/direct_vs_strategic_value.csv`
- `results/ppo_strategic_finetune/strategic_loss_summary.csv`
- `results/ppo_strategic_finetune/strategic_finetune_best.json`
- `results/ppo_training/ppo_capture_imitation.pt`
- `results/ppo_training/ppo_vs_strategic_finetuned.pt`
