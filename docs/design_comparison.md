# 설계 비교 문서

## 핵심 비교 질문

본 프로젝트는 윷놀이 전략 지식을 강화학습 agent에 주입하는 두 가지 방식을 비교한다.

1. 전략 지식을 **reward / imitation / policy prior**에 넣는 방식
2. 전략 지식을 **observation / state**에 넣는 방식

또한 초기 실험에서는 project-RF가 DQN / Dueling DQN을, RL-project가 PPO / MaskablePPO를 선택하여 서로 다른 강화학습 알고리즘을 비교하고자 했다.

## project-RF

project-RF는 처음에 DQN / Dueling DQN 기반 pure RL baseline을 실험했다. 그러나 sparse reward와 긴 episode 구조 때문에 학습이 불안정했고, 승률이 충분히 오르지 않았다.

이후 project-RF는 PPO 기반 모델에 전략 지식을 결합하는 방향으로 확장했다.

### 구성 요소

- PPO policy network
- DQN / Dueling DQN baseline
- engineered state
- capture-aware dense reward
- StrategicValue teacher imitation
- KL distillation
- inference-time tactical prior
- 최종 logits: `PPO logits + 2.5 * tactical bonus`

### State 특징

- 양측 말 위치
- pending yut result
- capture 가능 여부
- finish 가능 여부
- shortcut 가능 여부
- danger flag
- goal까지 남은 거리
- engineered / risk-aware encoder 실험

### Reward 설계

| 이벤트 | Reward |
| --- | ---: |
| 승리 | +100 |
| 패배 | -100 |
| 완주 | +30 |
| 잡기 | +35 |
| 잡힘 | -35 |
| 잡기 기회 놓침 | -10 |
| 위험 위치 | -15 |
| 위험 회피 | +5 |

### 장점

project-RF는 imitation learning과 tactical prior를 사용하여 상대적으로 적은 학습량에서도 잡기, 완주, 위험 회피 같은 전술적 행동을 보일 수 있었다.

### 한계

최종 Hybrid PPO는 pure PPO가 아니다. 최종 행동은 PPO network만으로 결정되지 않고, 사람이 설계한 tactical bonus가 logits에 추가된다.

## RL-project

RL-project는 sb3-contrib의 MaskablePPO를 중심으로 학습한 프로젝트이다. 전략 지식을 reward나 inference-time prior에 넣기보다는 observation feature에 넣는 방식을 사용했다.

### 구성 요소

- Gymnasium 기반 환경
- MaskablePPO
- tactical observation
- terminal reward
- action mask
- 40M step 장기 학습
- common paired evaluation protocol

### Observation 특징

- 기본 board state
- stack state
- yut pool
- action-level tactical feature
- legal flag
- capture count
- finish count
- move distance
- `rf_score`

### Reward 설계

| 이벤트 | Reward |
| --- | ---: |
| 승리 | +1 |
| 패배 | -1 |
| 그 외 | 0 |

### 장점

최종 action 선택은 PPO network가 수행한다. 즉, inference-time rule override 없이 observation에 포함된 tactical feature를 network가 학습해서 사용한다.

### 한계

sparse terminal reward는 학습에 많은 episode와 긴 training budget을 요구한다. 또한 학습 상대 분포와 seed에 따라 성능 편차가 생길 수 있다.

## 비교표

| 항목 | project-RF | RL-project |
| --- | --- | --- |
| 초기 알고리즘 | DQN / Dueling DQN | PPO / MaskablePPO |
| 최종 대표 모델 | Hybrid PPO | MaskablePPO |
| 지식 주입 위치 | reward, imitation, policy prior | observation/state |
| 최종 행동 선택 | PPO + tactical prior | PPO network only |
| Reward | dense tactical shaping | sparse terminal |
| 학습 효율 | prior 덕분에 빠른 전술 행동 가능 | 긴 학습량 필요 |
| Pure PPO 여부 | 아님 | action selection 기준으로 더 가까움 |
| 해석 | Hybrid RL | PPO + State Engineering |

## 해석

두 설계는 모두 강화학습 프로젝트로서 의미가 있지만, 답하려는 질문이 다르다.

- project-RF는 전략 지식을 imitation과 policy prior에 넣으면 실전 승률을 얼마나 높일 수 있는지를 보여준다.
- RL-project는 tactical feature를 observation으로 제공했을 때 PPO network가 전략을 얼마나 학습할 수 있는지를 보여준다.

공통 Rule-based baseline 상대에서는 두 agent가 거의 비슷한 성능을 보였다. 그러나 직접 대전에서는 project-RF Hybrid가 우세했고, tactical prior를 제거한 network-only 비교에서는 RL-project가 더 강했다.

따라서 최종 결론은 단순히 어느 agent가 더 강한지가 아니라, **평가 기준과 지식 주입 방식에 따라 성능 해석이 달라진다**는 것이다.
