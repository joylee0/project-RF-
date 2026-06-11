# Yutnori Reinforcement Learning Project

## Project Overview

본 프로젝트는 전통 게임인 **윷놀이**를 강화학습 환경으로 구현하고, 강화학습 agent가 윷놀이 전략을 학습할 수 있는지 분석하는 것을 목표로 한다.

특히 동일한 윷놀이 규칙과 공통 평가 프로토콜 하에서 두 가지 설계 철학을 비교하였다.

- **project-RF**: Hybrid PPO + Imitation Learning + Tactical Prior
- **RL-project**: MaskablePPO + Tactical Observation

핵심 비교 질문은 다음과 같다.

> 전략 지식을 reward/policy/prior에 넣는 방식과, observation/state에 넣는 방식은 성능에 어떤 차이를 만드는가?

---

## Reinforcement Learning Formulation

### State

- 내 말 위치
- 상대 말 위치
- 현재 사용 가능한 윷 결과
- 게임 진행 상태
- 프로젝트별 tactical feature

### Action

- 이동할 말 선택
- 사용할 윷 결과 선택
- action space: `4 pieces x 5 yut results = 20 actions`

### Reward

- 승리 / 패배 reward
- 프로젝트별 reward shaping 실험

### Environment

- Legacy No-Backdo Yutnori
- 2 players
- 4 pieces per player
- 잡기 / 업기 / 자동 지름길 / 윷·모 추가 턴
- action masking 적용

---

## Algorithms

본 프로젝트에서는 다음 알고리즘과 agent를 구현 및 실험하였다.

- PPO
- MaskablePPO
- DQN / Dueling DQN
- Value Network
- Strategic Value Agent
- Rule-based Agent

추가적으로 다음 기법을 실험하였다.

- Imitation Learning
- Tactical Prior
- Reward Shaping
- State Engineering
- Common Paired Evaluation

---

## Design Comparison

| 항목 | project-RF | RL-project |
| --- | --- | --- |
| 핵심 구조 | Hybrid PPO | MaskablePPO |
| 지식 주입 위치 | reward, imitation, tactical prior | observation/state |
| reward | capture-aware dense reward | terminal reward |
| action 선택 | PPO logits + tactical prior | PPO network only |
| 특징 | 전술 prior로 직접 대전 성능 강화 | network-only policy 구조가 명확함 |
| 한계 | pure PPO가 아닌 hybrid agent | 긴 학습량 필요 |

---

## Main Results

### Common Rule-based Evaluation

| Agent | Win Rate |
| --- | ---: |
| RL-project | 59.76% |
| project-RF | 59.46% |

### Head-to-Head Evaluation

| Matchup | Win Rate |
| --- | ---: |
| project-RF Hybrid | 53.98% |
| RL-project | 46.02% |

### Ablation

Tactical Prior 제거 시 project-RF 승률:

```text
17.87%
```

---

## Key Findings

- 두 접근 모두 Rule-based baseline 대비 약 60% 수준의 승률을 달성하였다.
- PPO 계열이 DQN 계열보다 안정적으로 학습되는 경향을 보였다.
- State Engineering과 Reward Design은 학습 성능에 큰 영향을 주었다.
- Tactical Prior는 project-RF의 직접 대전 성능 향상에 크게 기여하였다.
- 평가 기준에 따라 우수한 agent가 달라질 수 있으므로, 단일 승률만으로 일반적인 우위를 판단하기 어렵다.

---

## Repository Structure

```text
agents/
  PPO 및 tactical prior 기반 agent 구현

yut_rl/
  윷놀이 환경, state encoder, reward function, action encoding

train/
  PPO, DQN, config 기반 학습 스크립트

experiments/
  평가, ablation, common paired evaluation 스크립트

configs/
  실험 설정 파일

docs/
  보고서, 평가 프로토콜, 설계 비교 문서
```

---

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate common environment:

```bash
python experiments/validate_common_env.py
```

Train PPO:

```bash
python train/train_ppo.py --out-dir results/ppo_training
```

Train Pure Dueling DQN baseline:

```bash
python train/train_dueling_double_dqn.py \
  --episodes 20000 \
  --eval-every 1000 \
  --eval-games 500 \
  --out-dir results/pure_dueling_dqn
```

Run common paired evaluation:

```bash
python experiments/common_paired_evaluation.py \
  --my-agent ppo_capture_imitation \
  --friend-agent friend_ppo \
  --num-paired-seeds 2500 \
  --total-games 5000 \
  --output-dir results/common_paired_eval
```

---

## Report

자세한 설계, 실험 결과, 평가 프로토콜은 아래 문서를 참고한다.

- [Final Report Summary](docs/final_report_summary.md)
- [Design Comparison](docs/design_comparison.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [PPT Report Structure](docs/ppt_report_structure.md)
- [Result Report](result.md)

---

## Authors

- 남준우
- 이기쁨
