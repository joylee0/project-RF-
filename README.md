# Yutnori RL Project

## Project Overview

본 프로젝트는 전통 게임인 **윷놀이**를 강화학습 환경으로 구현하고, 강화학습 agent가 윷놀이 전략을 학습할 수 있는지 분석하는 것을 목표로 하는 프로젝트입니다.

초기 목표는 팀원별로 서로 다른 강화학습 모델을 선택하여 윷놀이 agent를 학습시키고 비교하는 것이었다.

- **이기쁨 / project-RF 초기 방향**: DQN / Dueling DQN 기반 value-based RL
- **남준우 / RL-project 초기 방향**: PPO / MaskablePPO 기반 policy-gradient RL

실험 과정에서 DQN 계열은 sparse reward와 긴 episode 구조에서 학습이 불안정하고 승률이 낮게 나타났다.  
이에 project-RF에서는 추가적으로 PPO 기반 모델에 전략 지식, imitation learning, tactical prior를 접목한 **Hybrid PPO Agent**를 실험하였다.

최종적으로 동일한 윷놀이 규칙과 공통 평가 프로토콜에서 다음 설계를 비교하였다.

- **project-RF**: Hybrid PPO + Imitation Learning + Tactical Prior
- **RL-project**: MaskablePPO + Tactical Observation

핵심 비교 질문은 다음과 같습니다.

> 전략 지식을 reward/policy/prior에 넣는 방식과, observation/state에 넣는 방식은 성능에 어떤 차이를 만드는가?

---

## Reinforcement Learning Formulation

### State

```text
- 내 말 위치
- 상대 말 위치
- 현재 사용 가능한 윷 결과
- 게임 진행 상태
- 프로젝트별 tactical feature
```

### Action

```text
- 이동할 말 선택
- 사용할 윷 결과 선택
- action space: `4 pieces x 5 yut results = 20 actions`
```

### Reward

```text
- 승리 / 패배 reward
- 프로젝트별 reward shaping 실험
```

### Environment

```text
- Legacy No-Backdo Yutnori
- 2 players
- 4 pieces per player
- 잡기 / 업기 / 자동 지름길 / 윷·모 추가 턴
- action masking 적용
```

---

## Algorithms

본 프로젝트에서는 다음 알고리즘과 agent를 구현 및 실험하였다.

- DQN / Dueling DQN: project-RF의 초기 pure RL baseline
- PPO: project-RF에서 추가로 학습한 PPO 계열 모델
- Hybrid PPO: project-RF의 최종 성능 개선 모델
- MaskablePPO: RL-project의 주요 학습 모델
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
| 초기 모델 | DQN / Dueling DQN | PPO / MaskablePPO |
| 최종 대표 모델 | Hybrid PPO | MaskablePPO |
| 지식 주입 위치 | reward, imitation, tactical prior | observation/state |
| reward | capture-aware dense reward | terminal reward |
| action 선택 | PPO logits + tactical prior | PPO network only |
| 특징 | 전술 prior로 직접 대전 성능 강화 | network-only policy 구조가 명확함 |
| 실험 흐름 | DQN 학습 한계 이후 PPO+전략 지식 접목 | 순수 RL 기반 PPO 장기 학습 |
| 한계 | pure PPO가 아닌 hybrid agent | 긴 학습량 필요 |

---

## Project Development Flow

1. 같은 윷놀이 환경에서 서로 다른 RL 알고리즘을 선택해 학습을 시작하였다.
2. project-RF는 DQN / Dueling DQN을 pure RL baseline으로 실험하였다.
3. DQN은 승률이 낮고 학습 안정성이 부족하여, 추가 개선 방향이 필요했다.
4. project-RF는 PPO 학습 모델을 추가하고, 여기에 StrategicValue teacher imitation과 tactical prior를 접목하였다.
5. RL-project는 MaskablePPO를 중심으로 tactical observation을 사용해 장기 학습하였다.
6. 최종 평가는 공통 환경과 paired evaluation protocol에서 수행하였다.

---

## Main Results

### project-RF Internal Development

| Model | Role |
| --- | --- |
| DQN / Dueling DQN | 초기 pure RL baseline, 최고 eval 승률 26.4% |
| PPO baseline | project-RF에서 추가 학습한 PPO 모델 |
| PPO imitation | StrategicValue teacher를 모방한 PPO |
| PPO capture imitation | capture 중심 imitation과 reward 설계를 적용한 PPO |
| PPO tactical | inference-time tactical prior를 적용한 Hybrid PPO |

### Common Rule-based Evaluation

| Agent      | Win Rate |
| ---------- | -------: |
| RL-project |   59.76% |
| project-RF |   59.46% |

### Head-to-Head Evaluation

| Matchup           | Win Rate |
| ----------------- | -------: |
| project-RF Hybrid |   53.98% |
| RL-project        |   46.02% |

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

## 실행 방법

필요한 패키지를 설치한다.

```bash
pip install -r requirements.txt
```

공통 윷놀이 환경이 정상적으로 동작하는지 검증한다.

```bash
python experiments/validate_common_env.py
```

PPO 모델을 학습한다.

```bash
python train/train_ppo.py --out-dir results/ppo_training
```

Pure Dueling DQN baseline을 학습한다.

```bash
python train/train_dueling_double_dqn.py \
  --episodes 20000 \
  --eval-every 1000 \
  --eval-games 500 \
  --out-dir results/pure_dueling_dqn
```

공통 paired evaluation을 실행한다.

```bash
python experiments/common_paired_evaluation.py \
  --my-agent ppo_capture_imitation \
  --friend-agent friend_ppo \
  --num-paired-seeds 2500 \
  --total-games 5000 \
  --output-dir results/common_paired_eval
```

---

## Trained Models

Deep learning 기반으로 학습한 모델 checkpoint는 별도 artifact로 저장한다.

다운로드 링크:

- [Trained model artifacts](./local_artifacts.zip)

주요 포함 모델:

```text
local_artifacts/results/ppo_training/ppo_baseline.pt
local_artifacts/results/ppo_training/ppo_capture_imitation.pt
local_artifacts/results/ppo_training/ppo_tactical.pt
local_artifacts/results/ppo_training/ppo_imitation.pt
local_artifacts/results/ppo_training/ppo_imitation_pretrained.pt
```

GitHub 업로드 시 `local_artifacts.zip` 파일이 크거나 repository에 포함하지 않는 경우, GitHub Releases 또는 Google Drive에 업로드한 뒤 위 링크를 다운로드 링크로 교체한다.

---

## Report

자세한 설계, 실험 결과, 평가 프로토콜은 아래 문서를 참고한다.

- [최종 보고서 요약](docs/final_report_summary.md)
- [설계 비교 문서](docs/design_comparison.md)
- [공통 평가 프로토콜](docs/evaluation_protocol.md)
- [Pure Dueling Double DQN baseline 결과](docs/pure_dueling_dqn_baseline_result.md)
- [프로젝트 결과 보고서](result.md)

---

## Authors

- 남준우
- 이기쁨
