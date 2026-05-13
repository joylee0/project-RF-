# 윷놀이 강화학습 프로젝트 가이드

이 프로젝트는 한국 전통 놀이인 윷놀이를 강화학습 환경으로 만들고, 여러 에이전트의 승률을 비교하기 위한 Python/PyTorch 프로젝트입니다.

현재 구현된 목표는 “완성된 최강 AI”가 아니라, 논문 방식에 가까운 모델과 기본 DQN 모델을 같은 환경에서 비교할 수 있는 실험 틀을 만드는 것입니다.

## 현재 구현 상태

- 2인 윷놀이
- 각 플레이어 말 4개
- 잡기 적용
- 업기 적용
- 윷/모 추가 던지기 적용
- 잡았을 때 추가 던지기 적용
- 뒷도 제외
- 지름길이 있는 기본 윷판 그래프 구현
- Random, Rule-based, DQN, Value Network 에이전트 구현
- 학습 중 승률 평가
- 모델 저장 및 이어서 학습 지원

## 윷 확률

현재 코드는 뒷도를 제외하고 아래 확률을 사용합니다.

- 도: 4/16
- 개: 6/16
- 걸: 4/16
- 윷: 1/16
- 모: 1/16

이 확률은 윷가락 네 개가 각각 같은 확률로 앞/뒤가 나온다고 가정한 기본 조합 확률입니다. 체감상 윷과 모가 자주 나오지 않는 점도 이 설정에 반영되어 있습니다.

## 에이전트 종류

### Random Agent

가능한 말 중 하나를 무작위로 선택합니다. 강화학습 모델이 최소한 이 에이전트보다 잘해야 합니다.

### Rule-based Agent

사람이 정한 간단한 규칙으로 행동합니다.

- 도착할 수 있으면 우선
- 상대 말을 잡을 수 있으면 우선
- 업힌 말 이동에 가산점
- 도착까지 남은 거리가 짧을수록 선호

### DQN Agent

상태를 입력으로 받고, 말 4개 각각에 대한 Q값을 출력합니다. 현재 상태에서 어느 말을 움직일지 직접 고르는 방식입니다.

### Value Network Agent

논문 방식에 더 가까운 기본 모델입니다.

가능한 행동을 하나씩 미리 시뮬레이션한 뒤, 각 행동으로 만들어지는 다음 상태를 신경망이 평가합니다. 그리고 다음 상태 가치가 가장 높은 행동을 선택합니다.

현재 기본 실행은 이 `Value Network Agent`를 사용합니다.

## 주요 파일

- `yut_rl/env.py`: 윷놀이 규칙과 환경
- `yut_rl/agents.py`: 에이전트와 신경망 모델
- `yut_rl/train.py`: 학습, 평가, 저장 실행 파일
- `requirements.txt`: 필요한 라이브러리
- `README.md`: GitHub용 프로젝트 소개
- `GUIDE_KO.md`: 한국어 사용 가이드

## 열지 않아도 되는 파일

아래 파일과 폴더는 Python과 PyTorch가 실행을 위해 자동으로 만든 내부 파일입니다.

- `.venv/`
- `.venv/bin/python`
- `.venv/bin/pip`
- `__pycache__/`
- `*.pyc`
- `checkpoints/*.pt`

특히 `.venv/bin/python`은 코드 파일이 아니라 Python 실행 프로그램입니다. 사람이 읽거나 수정하는 파일이 아닙니다.

## 설치

이미 `.venv`가 만들어져 있다면 다시 설치하지 않아도 됩니다.

처음부터 설치해야 한다면 아래 순서로 실행합니다.

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

기본값은 아래와 같습니다.

- 에이전트: `value`
- 에피소드: `50000`
- 평가 게임 수: `200`
- 중간 평가 간격: `5000`
- 저장 위치: `checkpoints/value.pt`

## 1000 에피소드로 테스트

```bash
.venv/bin/python -m yut_rl.train --episodes 1000 --eval-games 200 --eval-interval 200
```

## 기존 DQN 방식 실행

```bash
.venv/bin/python -m yut_rl.train --agent dqn --episodes 1000 --eval-games 200 --eval-interval 200
```

## 파라미터 바꾸기

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --eval-games 200 \
  --eval-interval 5000 \
  --lr 0.001 \
  --gamma 0.97 \
  --batch-size 64 \
  --epsilon-start 0.8 \
  --epsilon-end 0.05
```

주요 파라미터 의미:

- `--agent`: 사용할 모델, `value` 또는 `dqn`
- `--episodes`: 학습할 게임 수
- `--eval-games`: 승률 평가에 사용할 게임 수
- `--eval-interval`: 몇 에피소드마다 중간 승률을 출력할지
- `--lr`: 학습률
- `--gamma`: 미래 보상 반영 비율
- `--batch-size`: 한 번 업데이트할 때 사용할 샘플 수
- `--epsilon-start`: 학습 초반 무작위 행동 비율
- `--epsilon-end`: 학습 후반 최소 무작위 행동 비율

## 이어서 학습하기

```bash
.venv/bin/python -m yut_rl.train \
  --episodes 20000 \
  --load-model checkpoints/value.pt \
  --save-model checkpoints/value.pt
```

## GitHub에 올리는 방법

GitHub에는 코드와 문서만 올리고, `.venv`, `__pycache__`, `checkpoints` 같은 실행 결과물은 올리지 않습니다. `.gitignore`에 이미 제외 설정이 들어가 있습니다.

처음 올리는 경우:

```bash
cd /Users/joylee/Desktop/프로젝트
git init
git add .
git commit -m "Initial yutnori reinforcement learning project"
git branch -M main
git remote add origin https://github.com/사용자이름/저장소이름.git
git push -u origin main
```

이미 GitHub 저장소와 연결되어 있다면:

```bash
git add .
git commit -m "Update guide and value network training"
git push
```

GitHub에서 먼저 빈 저장소를 만든 뒤, 위 명령어의 `사용자이름/저장소이름` 부분만 본인 저장소 주소로 바꾸면 됩니다.

