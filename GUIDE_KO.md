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
- 말 위치 one-hot 상태 표현 적용
- 윷/모가 나오면 추가 던지기를 모두 끝낸 뒤 결과 묶음을 보관하고 원하는 순서로 사용
- 윷/모 연속 보너스는 최대 4회까지 허용
- 행동을 `사용할 윷 결과 + 움직일 말` 조합으로 확장
- 패배 시 `-1` 보상 적용
- DQN과 Value Network hidden size 기본값 256
- Value Network self-play에서 Random, Rule-based, 과거 스냅샷 모델을 섞어 상대
- Random, Rule-based, Tabular Q-learning, DQN, Value Network 에이전트 구현
- 학습 중 승률 평가
- 모델 저장 및 이어서 학습 지원

## 윷 확률

현재 코드는 논문에서 사용한 윷 확률을 기준으로 합니다.

- 도: 15.3% (논문의 도 11.5% + 아직 미구현된 뒷도 3.8%를 도로 처리)
- 개: 34.6%
- 걸: 34.6%
- 윷: 12.0%
- 모: 2.6%

논문 표의 확률은 반올림값이라 전체 합이 정확히 100%가 아닙니다. 현재 코드에는 뒷도가 아직 구현되어 있지 않기 때문에, 뒷도 확률은 도에 합쳐서 처리하고 도/개/걸/윷/모 확률의 합이 1이 되도록 정규화해서 사용합니다.

## 에이전트 종류

### Random Agent

가능한 말 중 하나를 무작위로 선택합니다. 강화학습 모델이 최소한 이 에이전트보다 잘해야 합니다.

### Rule-based Agent

사람이 정한 간단한 규칙으로 행동합니다.

- 도착할 수 있으면 우선
- 상대 말을 잡을 수 있으면 우선
- 업힌 말 이동에 가산점
- 도착까지 남은 거리가 짧을수록 선호

### Tabular Q-learning Agent

PyTorch 없이 Python 기본 기능만으로 실행되는 첫 강화학습 모델입니다.

상태별 Q값을 딕셔너리에 저장하고, 가능한 행동 중 Q값이 높은 행동을 선택합니다. 현재 행동은 단순히 말 4개 중 하나가 아니라 `윷 결과 5종 x 말 4개` 조합입니다.

### DQN Agent

one-hot 상태를 입력으로 받고, 최대 20개 행동에 대한 Q값을 출력합니다. 현재 보유한 윷 결과와 움직일 말을 함께 고르는 방식입니다.

### Value Network Agent

논문 방식에 더 가까운 기본 모델입니다.

가능한 행동을 하나씩 미리 시뮬레이션한 뒤, 각 행동으로 만들어지는 다음 상태를 신경망이 평가합니다. 그리고 다음 상태 가치가 가장 높은 행동을 선택합니다.

학습 중에는 현재 모델 혼자만 상대하지 않고, Random, Rule-based, 과거에 저장한 자기 자신의 스냅샷 모델을 섞어서 상대합니다. 이렇게 하면 한 가지 상대에게만 과적합되는 문제를 줄일 수 있습니다.

현재 기본 실행은 논문 방식에 더 가까운 `Value Network Agent`를 사용합니다.

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
- 에피소드: `1000`
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

## Value Network 방식 실행

PyTorch 설치가 되어 있을 때 실행할 수 있습니다.

```bash
.venv/bin/python -m yut_rl.train --agent value --episodes 1000 --eval-games 200 --eval-interval 200
```

스냅샷 상대 풀을 조절하려면 아래 옵션을 사용합니다.

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --snapshot-interval 1000 \
  --opponent-pool-size 5 \
  --rule-opponent-weight 4.0
```

## 파라미터 바꾸기

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --eval-games 200 \
  --eval-interval 5000 \
  --gamma 0.97 \
  --hidden-dim 256 \
  --epsilon-start 0.8 \
  --epsilon-end 0.05
```

주요 파라미터 의미:

- `--agent`: 사용할 모델, `tabular`, `value`, `dqn`
- `--episodes`: 학습할 게임 수
- `--eval-games`: 승률 평가에 사용할 게임 수
- `--eval-interval`: 몇 에피소드마다 중간 승률을 출력할지
- `--alpha`: Tabular Q-learning 학습률
- `--lr`: 학습률
- `--gamma`: 미래 보상 반영 비율
- `--hidden-dim`: DQN, Value Network의 은닉층 크기
- `--batch-size`: 한 번 업데이트할 때 사용할 샘플 수
- `--epsilon-start`: 학습 초반 무작위 행동 비율
- `--epsilon-end`: 학습 후반 최소 무작위 행동 비율
- `--snapshot-interval`: 몇 에피소드마다 현재 Value Network를 상대 모델로 저장할지
- `--opponent-pool-size`: 과거 스냅샷 상대를 몇 개까지 유지할지
- `--rule-opponent-weight`: Value Network 학습에서 Rule-based 상대를 얼마나 자주 만날지
- `--self-opponent-weight`: 현재 자기 자신을 상대할 가중치
- `--random-opponent-weight`: Random 상대 가중치
- `--snapshot-opponent-weight`: 과거 스냅샷 상대 가중치

## 저장된 모델 호환성

one-hot 상태 표현과 행동 공간이 바뀌었기 때문에, 예전에 저장한 `value.pt`나 `dqn.pt`는 새 모델 구조와 맞지 않을 수 있습니다. 구조 변경 후에는 새로 학습한 체크포인트를 사용하는 것이 좋습니다.

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
