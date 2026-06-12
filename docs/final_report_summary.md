# 최종 보고서 요약

## 프로젝트 주제

본 프로젝트는 한국 전통 놀이인 윷놀이를 강화학습 문제로 정의하고, 서로 다른 설계 철학을 가진 agent를 학습 및 평가한 프로젝트이다.

초기에는 팀원별로 서로 다른 강화학습 알고리즘을 선택해 실험을 진행했다.

1. **project-RF**: DQN / Dueling DQN 기반 value-based RL을 먼저 시도
2. **RL-project**: PPO / MaskablePPO 기반 policy-gradient RL을 중심으로 학습

실험 과정에서 project-RF의 DQN 계열 모델은 학습 안정성과 승률 측면에서 한계가 있었다. 이후 project-RF는 PPO 기반 모델에 StrategicValue teacher imitation과 tactical prior를 결합한 Hybrid PPO agent를 추가로 실험했다.

최종 비교는 공통 윷놀이 환경과 paired evaluation protocol을 기준으로 진행했다.

## 강화학습 문제 정의

- **State**: 말 위치, 상대 말 위치, 윷 결과, stack 정보, 게임 진행 상태
- **Action**: 이동할 말과 사용할 윷 결과 선택
- **Reward**: 승리/패배 보상 또는 프로젝트별 shaped reward
- **Policy**: 상태를 입력받아 행동을 선택하는 함수
- **Environment**: 윷놀이 규칙, 잡기, 업기, 지름길, 추가 턴, 종료 조건을 제공하는 시스템

## 설계 요약

| 항목 | project-RF | RL-project |
| --- | --- | --- |
| 초기 모델 | DQN / Dueling DQN | PPO / MaskablePPO |
| 최종 대표 모델 | Hybrid PPO | MaskablePPO |
| 지식 주입 위치 | reward, imitation, tactical prior | observation/state |
| State | engineered state | tactical observation |
| Reward | capture-aware dense reward | terminal reward |
| 최종 action 선택 | PPO logits + tactical prior | PPO network only |
| 분류 | Hybrid RL | PPO + State Engineering |

## 주요 결과

### Common Rule-based Evaluation

| Agent | Win Rate |
| --- | ---: |
| RL-project | 59.76% |
| project-RF | 59.46% |

두 agent 모두 공통 Rule-based baseline을 상대로 약 60% 수준의 승률을 보였다.

### Head-to-Head Evaluation

| Matchup | Win Rate |
| --- | ---: |
| project-RF Hybrid | 53.98% |
| RL-project | 46.02% |

직접 대전에서는 project-RF Hybrid agent가 우세했다.

### Tactical Prior Ablation

| 설정 | Win Rate |
| --- | ---: |
| project-RF tactical prior 제거 | 17.87% |
| RL-project network-only 기준 | 82.13% |

이 결과는 project-RF의 직접 대전 우위가 PPO network 자체만의 성능이라기보다, inference-time tactical prior의 영향이 크다는 것을 보여준다.

## 핵심 해석

- DQN / Dueling DQN은 초기 pure RL baseline으로 의미가 있었지만, sparse reward와 긴 episode 구조에서 학습이 어려웠다.
- PPO 계열은 윷놀이 환경에서 더 안정적인 학습 경향을 보였다.
- project-RF는 PPO에 imitation learning과 tactical prior를 결합하면서 직접 대전 성능을 높였다.
- RL-project는 tactical observation을 사용하되, 최종 action 선택은 PPO network가 수행한다.
- 평가 기준에 따라 우수한 agent가 달라질 수 있으므로, 단일 승률만으로 일반적인 우위를 판단하기 어렵다.

## 한계

- 두 프로젝트의 학습 budget이 동일하지 않다.
- project-RF의 최종 Hybrid PPO는 pure PPO가 아니다.
- RL-project는 장기 학습과 다중 seed를 사용했지만, project-RF는 일부 대표 checkpoint 중심으로 분석되었다.
- 주된 비교는 legacy no-backdo 환경을 기준으로 한다.

## 결론

본 프로젝트는 윷놀이 강화학습에서 알고리즘 자체뿐 아니라 state 설계, reward 설계, imitation learning, tactical prior가 성능에 큰 영향을 준다는 점을 보여준다.

project-RF는 정책과 prior에 전략 지식을 넣는 방식의 장단점을 보여주었고, RL-project는 observation에 전략 정보를 넣고 PPO network가 이를 학습하는 방식을 보여주었다.
