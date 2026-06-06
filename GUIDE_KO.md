# 윷놀이 강화학습 프로젝트 가이드

이 프로젝트는 한국 전통 놀이인 윷놀이를 강화학습 환경으로 만들고, 여러 에이전트의 성능을 비교하기 위한 Python/PyTorch 프로젝트입니다.

현재 프로젝트의 핵심은 다음 두 가지입니다.

- 윷놀이 환경을 코드로 구현하고 여러 강화학습 모델을 같은 조건에서 비교
- `strategic_value`를 강한 기준선으로 두고 PPO 계열 에이전트를 개선

전체 실험 결과 보고서는 [result.md](result.md)에 정리되어 있습니다.

## 현재 구현 상태

- 2인 윷놀이
- 각 플레이어 말 4개
- 잡기 적용
- 업기 적용
- 윷/모 추가 던지기 적용
- 잡았을 때 추가 던지기 적용
- 지름길이 있는 기본 윷판 그래프 구현
- 말 위치 one-hot 상태 표현
- 윷/모 결과를 모아두고 원하는 순서로 사용
- 윷/모가 연속 20번 나오면 즉시 승리
- PPO action masking 적용
- capture-aware PPO, StrategicValue 전용 fine-tuning 실험 추가

아직 뒷도는 별도 규칙으로 구현하지 않았습니다. 현재는 뒷도 확률을 도 확률에 합쳐서 처리합니다.

## 윷 확률

현재 코드는 논문에서 사용한 윷 확률을 기준으로 합니다.

| 결과 | 이동 칸 수 | 확률 |
| --- | ---: | ---: |
| 도 | 1 | 15.3% |
| 개 | 2 | 34.6% |
| 걸 | 3 | 34.6% |
| 윷 | 4 | 12.0% |
| 모 | 5 | 2.6% |

윷/모가 나오면 바로 행동하지 않고, 추가 던지기가 끝날 때까지 결과를 모아둔 뒤 보유한 결과 중 하나를 선택해 사용합니다.

## 주요 파일

```text
yut_rl/
  env.py       # 윷놀이 규칙, 윷판 이동, 환경 상태
  agents.py    # 기본 에이전트, DQN, Value Network, PPO 계열
  train.py     # 기본 학습 및 평가 실행 파일

agents/
  ppo_agent.py # Masked PPO, Capture-aware PPO

train/
  train_ppo.py # PPO imitation, curriculum, capture-aware 학습

experiments/
  tournament.py                # 에이전트 tournament
  improve_agents.py            # strategic_value/DQN 개선 실험
  evaluate_ppo.py              # PPO 평가 및 그래프 저장
  strategic_ppo_finetune.py    # strategic_value 전용 PPO fine-tuning

results/
  ppo_eval/
  ppo_training/
  ppo_strategic_finetune/

result.md      # 현재까지의 실험 결과 보고서
README.md      # 프로젝트 개요
GUIDE_KO.md    # 한국어 사용 가이드
```

## 열지 않아도 되는 파일

아래 파일과 폴더는 Python과 PyTorch 실행 과정에서 자동으로 만들어지거나 사람이 직접 수정하지 않는 파일입니다.

- `.venv/`
- `.venv/bin/python`
- `.venv/bin/pip`
- `__pycache__/`
- `*.pyc`
- `*.pt`

특히 `.venv/bin/python`은 코드 파일이 아니라 Python 실행 프로그램입니다.

## 설치

이미 `.venv`가 만들어져 있다면 다시 설치하지 않아도 됩니다.

처음 설치하는 경우:

```bash
cd /Users/joylee/Desktop/프로젝트
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 기본 실행

```bash
cd /Users/joylee/Desktop/프로젝트
.venv/bin/python -m yut_rl.train
```

기본값:

- 에이전트: `value`
- 에피소드: `1000`
- 평가 게임 수: `200`
- 저장 위치: `checkpoints/value.pt`

## 빠른 테스트

```bash
.venv/bin/python -m yut_rl.train --episodes 20 --eval-games 20 --eval-interval 0
```

## 1000 에피소드 실험

```bash
.venv/bin/python -m yut_rl.train --episodes 1000 --eval-games 200 --eval-interval 200
```

## 기본 에이전트 실행

```bash
.venv/bin/python -m yut_rl.train --agent strategic --eval-games 200
.venv/bin/python -m yut_rl.train --agent value --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent strategic-value --episodes 1000 --eval-games 200
```

## DQN 계열 실행

```bash
.venv/bin/python -m yut_rl.train --agent dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent double-dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent dueling-dqn --episodes 1000 --eval-games 200
```

## Policy Gradient 계열 실행

```bash
.venv/bin/python -m yut_rl.train --agent reinforce --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent a2c --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent ppo --episodes 1000 --eval-games 200
```

## PPO 개선 실험 실행

PPO imitation, curriculum, capture-aware 학습:

```bash
.venv/bin/python train/train_ppo.py \
  --out-dir results/ppo_training \
  --eval-games 1000
```

Capture-aware PPO만 학습:

```bash
.venv/bin/python train/train_ppo.py \
  --train-capture-agents \
  --out-dir results/ppo_training \
  --eval-games 1000 \
  --capture-samples 8000 \
  --capture-imitation-epochs 8
```

PPO tournament 평가:

```bash
.venv/bin/python experiments/evaluate_ppo.py \
  --games 1000 \
  --model-dir results/ppo_training \
  --out-dir results/ppo_eval
```

StrategicValue 전용 PPO fine-tuning:

```bash
.venv/bin/python experiments/strategic_ppo_finetune.py \
  --analysis-games 1000 \
  --eval-games 1000 \
  --out-dir results/ppo_strategic_finetune
```

## 에이전트 설명

### Random Agent

가능한 행동 중 하나를 무작위로 선택합니다.

### Rule-based Agent

사람이 정한 간단한 규칙으로 행동합니다.

- 도착할 수 있으면 우선
- 상대 말을 잡을 수 있으면 우선
- 업힌 말 이동에 가산점
- 도착까지 남은 거리가 짧을수록 선호

### Strategic Rule-based Agent

기존 Rule-based보다 윷놀이 전략을 더 많이 반영한 기준선입니다.

각 행동을 복사한 환경에서 시뮬레이션한 뒤 완주, 잡기, 진행도, 업힌 말 이동, 지름길 진입, 상대에게 잡힐 위험, 상대의 다음 반격 가능성을 함께 점수화합니다.

### Value Network Agent

가능한 행동을 하나씩 시뮬레이션한 뒤, 각 행동으로 만들어지는 다음 상태를 신경망이 평가합니다. 논문 방식에 가까운 기본 강화학습 모델입니다.

### Strategic Value Network Agent

Value Network의 예측 점수와 Strategic Rule-based의 전략 점수를 섞어서 행동을 고르는 개선 모델입니다.

현재 실험에서 가장 강한 기준선 역할을 합니다.

### PPO Agent

정책이 한 번에 너무 크게 바뀌지 않도록 clipped objective를 사용하는 policy gradient 계열 모델입니다.

프로젝트 후반부에서는 PPO에 다음 개선을 추가했습니다.

- legal action masking
- StrategicValue teacher imitation
- teacher score distillation
- capture-aware reward shaping
- tactical state oversampling
- StrategicValue 전용 fine-tuning
- counterplay/danger 기반 tactical prior

## 현재 주요 결과

PPO tournament 결과:

| Agent | 전체 승률 | 평균 잡기 |
| --- | ---: | ---: |
| strategic_value | 72.86% | 2.81 |
| strategic_rule | 72.19% | 2.55 |
| ppo_capture_imitation | 61.76% | 2.77 |
| ppo_tactical | 61.01% | 2.78 |
| ppo_imitation | 41.16% | 1.33 |

StrategicValue 직접 대결:

| Agent | StrategicValue 상대 승률 | 평균 잡기 |
| --- | ---: | ---: |
| ppo_capture_imitation | 46.0% | 1.84 |
| ppo_vs_strategic_finetuned | 47.6% | 1.88 |

자세한 결과는 [result.md](result.md)를 참고하면 됩니다.

## 결과 파일

- `results/ppo_eval/ppo_tournament_summary.csv`
- `results/ppo_eval/ppo_tournament_matchups.csv`
- `results/ppo_eval/best_ppo.json`
- `results/ppo_strategic_finetune/direct_vs_strategic_value.csv`
- `results/ppo_strategic_finetune/strategic_loss_summary.csv`
- `results/ppo_strategic_finetune/strategic_finetune_best.json`

## 파라미터 의미

- `--agent`: 사용할 모델
- `--episodes`: 학습 게임 수
- `--eval-games`: 평가 게임 수
- `--eval-interval`: 중간 평가 간격
- `--lr`: 학습률
- `--gamma`: 미래 보상 반영 비율
- `--hidden-dim`: 신경망 은닉층 크기
- `--batch-size`: 한 번 업데이트할 때 사용할 샘플 수
- `--epsilon-start`: 학습 초반 무작위 행동 비율
- `--epsilon-end`: 학습 후반 최소 무작위 행동 비율
- `--lookahead-depth`: Value Network가 몇 단계까지 다음 행동을 볼지
- `--heuristic-weight`: Strategic Value Network에서 전략 점수를 얼마나 섞을지
- `--rollout-episodes`: policy gradient 업데이트 전 모을 게임 수
- `--ppo-epochs`: PPO 업데이트 반복 횟수
- `--ppo-clip`: PPO clipping 범위
- `--mcts-simulations`: MCTS + Value 평가 시 행동당 시뮬레이션 수

## 현재 한계

- 뒷도는 아직 별도 규칙으로 구현하지 않음
- 실제 공식 윷놀이 규칙 전체를 완전히 재현한 것은 아님
- PPO 개선 모델은 순수 신경망만이 아니라 tactical prior도 함께 사용함
- 결과는 seed, 평가 게임 수, 학습 길이에 따라 달라질 수 있음
