# 윷놀이 강화학습 프로젝트 최종 보고서

## 1. 프로젝트 주제 및 목표

본 프로젝트의 주제는 한국 전통 놀이인 윷놀이를 강화학습 문제로 정의하고, 고정된 Rule-based Agent를 상대하는 PPO 기반 agent를 학습하여 승률을 높이는 것이다.

윷놀이는 확률적인 윷 결과와 전략적인 말 선택이 결합된 게임이다. 따라서 정적 데이터셋을 사용하는 지도학습이 아니라, 게임 규칙을 구현한 시뮬레이션 환경에서 episode를 생성하고 agent가 반복적인 상호작용을 통해 정책을 학습하도록 설계했다.

본 프로젝트에서는 PPO를 주요 강화학습 모델로 사용했다. 내가 설계한 Rule-based Agent는 학습 대상이 아니라 PPO가 이겨야 하는 고정 opponent이자 baseline이며, 팀원의 최종 목표는 이 Rule-based Agent를 상대로 약 60% 승률을 달성하는 것이었다. 최종적으로는 팀원 프로젝트와 동일한 **common env paired evaluation**, 즉 같은 환경과 같은 seed에서 선공/후공을 바꿔 평가하는 방식으로 PPO와 Rule-based Agent의 승률을 평가했다.

최종 비교의 핵심 질문은 다음과 같다.

- 같은 윷놀이 규칙과 같은 seed 조건에서 PPO와 고정 Rule-based Agent를 공정하게 비교할 수 있는가?
- Rule-based Agent를 기준선으로 둘 때 PPO가 얼마나 높은 승률을 학습하는가?
- 선공/후공 편향을 줄인 paired evaluation에서 실제 승률은 어떻게 달라지는가?

본 프로젝트의 핵심 기여는 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| 문제 정의 | 윷놀이를 2인 확률 기반 보드게임 강화학습 문제로 정의 |
| 환경 구현 | 윷 확률, 잡기, 업기, 지름길, HOME 처리, 추가 턴 구현 |
| 기준 agent | 고정 Rule-based Agent를 PPO 학습 및 평가 opponent로 사용 |
| 학습 모델 | action masking을 적용한 PPO 계열 모델 사용 |
| 평가 방식 | seed와 선공/후공을 통제한 common paired evaluation |

## 2. 환경 및 데이터셋 설명

### 2.1 데이터셋 대신 시뮬레이션 환경 사용

본 프로젝트에는 외부 정적 데이터셋이 존재하지 않는다. 대신 윷놀이 게임 엔진을 직접 구현하고, agent가 환경과 상호작용하면서 학습 episode를 생성한다.

```text
Dataset 대신 사용한 것:
- self-play 및 opponent-play 기반 simulated episode
- seed로 재현 가능한 윷 결과 sequence
- 매 decision step마다 생성되는 state, action, reward, done 정보
```

강화학습 관점에서 하나의 episode는 윷놀이 한 판에 해당한다. episode는 한 플레이어의 말 4개가 모두 `FINISHED` 상태가 되면 종료된다.

### 2.2 환경 규칙

윷놀이는 확률적 요소와 전략적 의사결정이 동시에 존재한다. 따라서 환경은 다음 요소를 포함하도록 설계했다.

- 2인 대전
- 플레이어당 말 4개
- 잡기와 업기
- 윷/모 추가 턴
- 잡기 후 추가 턴
- 지름길 자동 진입
- HOME 처리
- legal action mask
- seed 기반 재현성

최종 common env 기준에서는 다음 특수 규칙을 제외했다.

- 뒷도
- 낙
- 후진 이동
- 플레이어의 지름길 선택
- 윷/모 20회 연속 즉시 승리

### 2.3 윷 확률

모든 최종 평가에서 사용한 확률은 다음과 같다.

| 결과 | 이동량 | 확률 |
| --- | ---: | ---: |
| 도 | 1 | 0.1536 |
| 개 | 2 | 0.3456 |
| 걸 | 3 | 0.3456 |
| 윷 | 4 | 0.1296 |
| 모 | 5 | 0.0256 |

### 2.4 Seed와 Episode 생성 방식

평가는 윷 결과의 무작위성과 선공 이점이 승률에 미치는 영향을 줄이기 위해 paired seed 방식으로 수행했다. 하나의 base seed마다 선공과 후공을 바꾼 두 게임을 실행한다.

```text
base seed 1개당 2게임 생성

Game A: PPO 선공, Rule-based Agent 후공
Game B: Rule-based Agent 선공, PPO 후공

paired seed 수: 2,500
PPO seed별 평가 게임 수: 5,000
3개 학습 seed 기준 총 평가 게임 수: 15,000
```

이 방식은 같은 확률 환경에서 두 agent가 모두 선공과 후공을 경험하게 하므로, 선공 편향을 줄이고 seed별 분산을 확인할 수 있다.

### 2.5 Preprocessing

환경에서 생성된 원본 게임 상태는 PPO가 사용할 수 있는 vector observation으로 변환된다. 주요 preprocessing은 다음과 같다.

| 단계 | 설명 |
| --- | --- |
| 위치 인코딩 | 각 말의 위치와 상태를 숫자 또는 one-hot 형태로 변환 |
| stack 정보 인코딩 | 같은 칸에 업힌 말들을 stack matrix로 표현 |
| pool 인코딩 | 현재 턴에서 아직 사용하지 않은 윷 결과 개수를 5차원 vector로 표현 |
| action mask 생성 | 현재 상태에서 선택 가능한 action만 `True`로 표시 |
| tactical feature 생성 | 각 action의 capture, finish, distance, rf_score 등을 계산 |

이 preprocessing은 raw game state를 강화학습 모델이 학습 가능한 고정 길이 vector로 변환하는 과정이다.

## 3. State / Action / Reward 설계

### 3.1 Action 설계

행동은 말 선택과 현재 사용할 윷 결과를 결합한 형태로 정의했다.

```text
action = piece_id * 5 + yut_id

piece_id: 0~3
yut_id: 도, 개, 걸, 윷, 모 = 0~4
총 action 수: 4 pieces x 5 yut results = 20
```

불가능한 행동은 legal action mask로 제외했다. PPO 계열 agent는 policy logits에서 illegal action을 `-1e9`로 masking하여 잘못된 행동을 선택하지 않도록 했다.

예를 들어 현재 pool에 `도`가 없거나, 이미 완주한 말을 선택하는 action은 mask에서 제외된다. 이 action masking은 윷놀이처럼 매 상태마다 가능한 행동이 달라지는 환경에서 필수적이다.

### 3.2 Base State

base state는 게임 규칙에서 직접 관측 가능한 기본 정보로 구성된다.

| 정보 | 설명 |
| --- | --- |
| 내 말 위치 | 4개 말의 위치와 상태 |
| 상대 말 위치 | 상대 4개 말의 위치와 상태 |
| stack 정보 | 같은 칸에 업힌 말 정보 |
| pool count | 현재 턴에서 아직 사용하지 않은 윷 결과 |
| 현재 player | 현재 action을 선택해야 하는 player |
| legal action mask | 현재 선택 가능한 action |

base state는 가장 기본적인 표현이지만, capture나 finish 가능성을 모델이 직접 추론해야 하므로 학습 난이도가 높다.

### 3.3 Tactical State

tactical state는 base state에 사람이 설계한 action-level 전략 feature를 추가한 표현이다. 이 feature들은 내가 설계한 Rule-based Agent의 판단 기준과도 연결된다.

| Tactical feature | 의미 |
| --- | --- |
| `legal` | 해당 action이 합법인지 여부 |
| `capture` | 상대 말을 잡는 action인지 여부 |
| `captured_count` | 잡을 수 있는 상대 말 수 |
| `finish` | 완주가 발생하는 action인지 여부 |
| `finished_count` | 완주하는 내 말 수 |
| `moved_count` | stack 이동으로 함께 움직이는 말 수 |
| `waiting_move` | 대기 중인 말을 출발시키는 action인지 여부 |
| `stack_size` | 이동하는 stack 크기 |
| `distance_after` | 이동 후 완주까지 남은 거리 |
| `rf_score` | Rule-based Agent의 action 평가 점수 |

따라서 최종 TeamPPO는 원본 상태만 사용하는 Pure RL이라기보다, 윷놀이 도메인 지식을 state에 반영한 **RL + State Engineering** 모델로 분류하는 것이 적절하다. 다만 action을 rule로 강제로 바꾸지는 않고, 최종 행동 선택은 PPO policy가 수행한다.

### 3.4 Reward 설계

#### Terminal reward

최종 TeamPPO의 주요 학습 설정은 terminal reward다.

```text
승리: +1
패배: -1
그 외 step: 0
```

terminal reward는 최종 목표인 승패와 직접 일치한다. 하지만 보상이 episode 마지막에만 주어지기 때문에 credit assignment가 어렵다는 단점이 있다.

#### RF-shaped reward

비교 실험을 위해 Rule-based Agent의 전술 선호를 반영한 shaped reward도 구현했다.

```text
learner capture: +0.08 * captured_count
learner finish: +0.15 * finished_count
learner shortcut: +0.02
opponent capture: -0.08 * captured_count
opponent finish: -0.15 * finished_count
```

실험 결과, reward shaping은 capture 성향을 강화할 수 있었지만 전체 승률을 항상 개선하지는 않았다. 최종 후보에서는 `tactical state + terminal reward` 조합이 가장 안정적인 설정으로 선택되었다.

## 4. Agent와 Model

### Rule-based 계열

`RuleBasedAgent`는 완주, 잡기, 업기, 남은 거리 등을 기준으로 행동을 선택한다. 프로젝트에서 내 agent는 이 rule-based 계열을 기반으로 설계되었고, 팀원 프로젝트에서는 이를 `project_rf_rule` 형태로 포팅하여 사용했다.

이 agent는 학습 대상이라기보다 PPO 성능을 평가하는 고정 opponent 역할을 한다. 따라서 내 역할은 단순히 작은 baseline을 만든 것이 아니라, 강화학습 agent의 성능 기준과 난이도를 정의한 것이다.

### Strategic / Hybrid 계열

`StrategicRuleBasedAgent`와 `StrategicValueNetworkAgent`는 우리 로컬 프로젝트에서 사용한 더 강한 heuristic baseline이다.

- `StrategicRuleBasedAgent`: 행동별 시뮬레이션을 통해 완주, 잡기, 위험도, 상대 반격 가능성을 점수화
- `StrategicValueNetworkAgent`: value network 예측과 strategic heuristic score를 결합

이 계열은 MyAgent보다 강한 비교 기준이나 분석용 baseline으로 의미가 있다. 다만 최종 TeamPPO 40M 모델은 이 agent를 imitation learning teacher로 직접 사용한 모델이 아니므로, 최종 PPO 결과와는 분리해서 해석해야 한다.

### PPO 계열

PPO는 policy gradient 기반 모델이며, 윷놀이처럼 같은 상태에서도 여러 전략적 행동이 가능한 환경에 적합하다고 판단했다.

PPO 학습과 개선은 다음 방향으로 진행했다.

1. 기본 PPO
2. action masking PPO
3. tactical observation 적용
4. terminal reward와 rf_shaped reward 비교
5. 같은 환경과 같은 seed에서 선공/후공을 바꿔 평가하는 common paired evaluation

핵심은 PPO가 불가능한 action을 선택하지 않도록 masking하고, 잡기·완주·위험 회피처럼 윷놀이에 중요한 정보를 state와 reward에 반영하는 것이었다.

최종 TeamPPO 40M 학습 설정은 다음과 같다.

| 항목 | 설정 |
| --- | --- |
| Algorithm | MaskablePPO |
| Policy | MlpPolicy |
| Observation | tactical |
| Reward | terminal |
| Opponent | common_rule_based |
| Training seeds | 0, 1, 2 |
| Total timesteps | seed별 40M |
| Vector env | SubprocVecEnv |
| Number of envs | 12 |
| Device | CUDA |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| Batch size | 64 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Entropy coef | 0.0 |

하이퍼파라미터 튜닝은 단순히 learning rate를 바꾸는 방식보다, observation mode, reward mode, training budget, vectorized environment 설정을 단계적으로 조정하는 방식으로 진행했다.

## 5. 실험 설계

### Common Env 검증

팀원 프로젝트와 비교하기 위해 별도 common env 검증 스크립트를 작성했다.

검증 항목:

- board size
- yut probability
- action space
- legal action mask
- state dimension
- reward output
- terminal condition
- opponent behavior
- seed reproducibility

이를 통해 평가 전에 규칙과 인터페이스가 동일한지 확인했다.

### State / Reward 설계 비교와 Algorithm 확장 구조

강화학습 성능이 단순 모델 종류만이 아니라 state/reward/action 설계에 따라 달라진다는 점을 분석하기 위해 다음 실험 파일을 구성했다.

- `experiments/ablation_state_design.py`
- `experiments/ablation_reward_design.py`
- `experiments/compare_rl_algorithms_common_env.py`

로컬 프로젝트에서 확장 가능하도록 구성한 비교 대상은 다음과 같다.

- State: Raw, Board, Engineered, RiskAware
- Reward: Sparse, MinimalDense, BalancedTactical, RiskAware
- Algorithm: DQN, DoubleDQN, DuelingDQN, A2C, PPO, MaskedPPO

이 구조는 단순 최종 승률만 보는 것이 아니라 learning curve, sample efficiency, seed stability를 함께 보기 위한 실험 확장 구조이다. 실제 최종 보고의 핵심 실험은 팀원 PPO 프로젝트에서 수행한 state/reward 비교와 장기 학습 결과를 중심으로 해석한다.

팀원 PPO 프로젝트에서는 실제로 state/reward 조합을 바꿔 3M candidate sweep을 수행했다. 결과는 다음과 같다.

| Observation | Reward | 평균 승률 |
| --- | --- | ---: |
| base | terminal | 37.4% |
| base | rf_shaped | 32.4% |
| tactical | terminal | **53.3%** |
| tactical | rf_shaped | 52.3% |

이 결과에서 가장 큰 개선은 reward shaping이 아니라 state 설계에서 발생했다. `tactical` observation은 capture, finish, distance, rf_score 등 Rule-based Agent가 사용하는 판단 기준을 action-level feature로 제공했고, 이로 인해 PPO가 각 행동의 전술적 의미를 더 쉽게 학습할 수 있었다.

반면 `rf_shaped` reward는 잡기 성향을 강화하는 효과는 있었지만, 전체 승률에서는 `terminal` reward보다 낮았다. 따라서 장기 학습 후보는 `tactical + terminal` 조합으로 선택했다.

학습량과 평가 기준을 바꾼 장기 실험 흐름은 다음과 같다.

| 단계 | 설정 | 평균 승률 | 해석 |
| --- | --- | ---: | --- |
| 3M sweep | tactical + terminal | 53.3% | 장기 학습 후보 선정 |
| 10M long training | tactical + terminal | 57.95% | 학습량 증가로 성능 상승 |
| 30M common paired | tactical + terminal | 56.49% | 공통 평가 기준 적용 후 재측정 |
| 40M retraining | tactical + terminal + common opponent | 59.76% | 60% 목표에 근접 |

이를 통해 최종 성능은 단일 PPO 알고리즘만의 결과가 아니라, state 설계, reward 선택, action masking, 학습량, 평가 환경 정렬이 함께 작용한 결과임을 확인했다.

## 6. 최종 평가 방식

최종 평가는 **common paired evaluation**으로 수행했다.

평가 지표는 다음과 같다.

| 지표 | 의미 |
| --- | --- |
| 전체 승률 | 전체 평가 게임 중 PPO가 이긴 비율 |
| 선공 승률 | PPO가 선공인 게임에서의 승률 |
| 후공 승률 | PPO가 후공인 게임에서의 승률 |
| seed 평균 / 표준편차 | 학습 seed별 성능 안정성 |
| illegal action 수 | 모델이 불가능한 action을 선택한 횟수 |
| evaluation error 수 | 평가 중 비정상 종료 또는 오류 횟수 |

평가 방식:

```text
base seed 1개당 2게임 실행

Game A: TeamPPO 선공, MyAgent 후공
Game B: MyAgent 선공, TeamPPO 후공

paired seed 수: 2,500
총 게임 수: 5,000 per model
TeamPPO seed 수: 3개
총 평가 게임 수: 15,000
```

이 방식은 선공/후공 편향을 줄이고, 같은 seed 조건에서 두 agent가 모두 선공과 후공을 경험하도록 만든다.

## 7. 최종 평가 결과

최종 직접 비교는 팀원 PPO와 내 agent를 같은 common env에 올려 수행했다.

- TeamPPO: `David-Nam/RL-yutnori`의 `bests/ppo_common_rule_40m_subproc`
- MyAgent: 팀원 저장소에 포팅된 `project_rf_rule`
- observation mode: tactical
- policy: deterministic
- illegal action: 모두 0
- evaluation error: 모두 0

| TeamPPO seed | TeamPPO 승률 | MyAgent 승률 | TeamPPO 선공 | TeamPPO 후공 | MyAgent 선공 | MyAgent 후공 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 57.26% | 42.74% | 56.80% | 57.72% | 42.28% | 43.20% |
| 1 | 57.56% | 42.44% | 58.96% | 56.16% | 43.84% | 41.04% |
| 2 | 55.08% | 44.92% | 56.36% | 53.80% | 46.20% | 43.64% |

종합 결과:

```text
TeamPPO 평균 승률: 56.63%
MyAgent 평균 승률: 43.37%
TeamPPO pooled 승률: 56.63%
MyAgent pooled 승률: 43.37%
seed std: 1.11%p
총 게임 수: 15,000
illegal actions: 0
evaluation errors: 0
```

해석:

공통 env와 paired seed 조건에서 TeamPPO는 세 개 학습 seed 모두 MyAgent보다 높은 승률을 보였다. MyAgent는 강한 rule-based 기준선 역할을 했고, TeamPPO는 이 기준선을 상대로 약 56.6% 승률을 기록했다.

## 8. 참고 평가: TeamPPO vs Common Rule-Based

팀원 PPO가 공통 rule-based baseline을 상대로 어느 정도 성능을 보이는지도 재현했다.

| TeamPPO seed | 승률 | 선공 | 후공 | 통과 여부 |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 58.42% | 59.00% | 57.84% | false |
| 1 | 60.46% | 60.76% | 60.16% | true |
| 2 | 60.40% | 61.32% | 59.48% | true |

3-seed 평균 승률은 59.76%였다. 이 결과는 팀원의 60% 목표에 거의 도달한 참고 결과다. 다만 최종 직접 대결은 MyAgent(`project_rf_rule`) 상대 평가이므로, 최종 결과는 56.63%로 해석해야 한다.

## 9. 어려웠던 점과 해결 방법

### 1. 윷놀이 규칙의 변형이 많음

윷놀이는 지역·프로젝트마다 HOME 처리, 지름길, 업기, 잡기, 윷/모 추가 턴 규칙이 다를 수 있다.

해결 방법:

- 공통 규칙을 문서화했다.
- `validate_common_env.py`로 보드 크기, 확률, action mask, terminal condition을 검증했다.
- 평가 전용 common env를 따로 구성했다.

### 2. 프로젝트 목표와 평가 기준이 혼동될 수 있음

강화학습 프로젝트에서는 단순히 여러 agent를 나열해 비교하는 것보다, 어떤 opponent를 기준으로 어떤 모델을 학습했는지 명확히 하는 것이 중요하다. 이 프로젝트의 기준은 고정 Rule-based Agent를 상대하는 PPO 모델이다.

해결 방법:

- Rule-based Agent를 고정 opponent이자 baseline으로 정의했다.
- PPO를 주요 학습 모델로 정리했다.
- 최종 평가는 common paired evaluation 기준으로만 해석했다.

### 3. 선공/후공 편향

윷놀이는 선공 여부가 승률에 영향을 줄 수 있다.

해결 방법:

- base seed마다 두 게임을 실행했다.
- 같은 seed에서 선공과 후공을 교대했다.
- 선공 승률, 후공 승률, agent별 선공/후공 승률을 따로 기록했다.

### 4. 서로 다른 프로젝트의 모델을 직접 비교하기 어려움

팀원 PPO는 `MaskablePPO model.zip` 형식이고, 내 프로젝트 PPO는 PyTorch `.pt` 형식이었다. 또한 observation dimension과 env wrapper가 달랐다.

해결 방법:

- 팀원 저장소의 common env와 `project_rf_rule` 포팅 구현을 기준으로 직접 대결 평가 스크립트를 작성했다.
- `team_ppo_vs_my_agent_common_eval.py`에서 팀원 `model.zip`을 로드하고, MyAgent는 `project_rf_rule`로 실행했다.

### 5. 모델 파일과 결과 파일 관리

학습 checkpoint, CSV, PNG가 많아 GitHub에 올리기 어려웠다.

해결 방법:

- `results/`, `checkpoints/`, `local_artifacts/`를 `.gitignore`로 제외했다.
- 최종 결과는 `docs/`의 Markdown 보고서로 정리했다.
- 로컬 artifact는 `local_artifacts/`에 보관했다.

## 10. 프로젝트 산출물

주요 코드:

- `common_rule_based_env.py`
- `experiments/team_ppo_vs_my_agent_common_eval.py`
- `experiments/common_paired_evaluation.py`
- `experiments/validate_common_env.py`
- `yut_rl/state_encoders.py`
- `yut_rl/reward_functions.py`
- `yut_rl/action_encodings.py`
- `yut_rl/config_runner.py`

주요 문서:

- `README.md`
- `result.md`
- `docs/team_ppo_vs_my_agent_common_eval.md`
- `docs/team_ppo_common_rule_eval.md`
- `docs/state_reward_action_analysis.md`

## 11. 최종 결론

이 프로젝트는 단순히 PPO 승률을 높이는 것보다, 윷놀이 강화학습 실험에서 **공정한 평가 기준을 만드는 과정**이 중요했다.

내 agent는 rule-based opponent로서 PPO 학습과 평가의 기준선 역할을 했다. 팀원 PPO는 같은 환경과 같은 seed에서 선공/후공을 바꿔 평가한 common paired evaluation에서 MyAgent를 상대로 평균 56.63% 승률을 기록했고, 공통 `common_rule_based` 참고 기준에서는 평균 59.76%로 60% 목표에 거의 도달했다.

최종적으로 다음을 확인했다.

- 윷놀이 강화학습은 state/reward/action 설계에 민감하다.
- PPO는 action masking과 tactical observation을 사용할 때 안정적으로 성능이 향상된다.
- paired seed 기반 common evaluation은 선공/후공 편향을 줄이는 데 도움이 된다.
- MyAgent는 PPO 성능을 검증하는 기준선으로 의미가 있으며, TeamPPO는 이 기준선을 넘어서는 전략을 학습했다.
