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
- 윷/모가 연속 20번 나오면 즉시 승리
- 행동을 `사용할 윷 결과 + 움직일 말` 조합으로 확장
- 패배 시 `-1` 보상 적용
- 기본 보상 방식은 `hybrid`: 최종 승패 보상과 중간 보상을 함께 사용
- DQN과 Value Network hidden size 기본값 256
- DQN 계열은 상대 턴으로 넘어간 뒤 내 차례가 돌아온 상태를 다음 상태로 학습
- Value Network self-play에서 Random, Rule-based, 과거 스냅샷 모델을 섞어 상대
- Random, Rule-based, Strategic Rule-based, Tabular Q-learning, DQN, Double DQN, Dueling DQN, REINFORCE, A2C, PPO, Value Network, Strategic Value Network, MCTS + Value Network 에이전트 구현
- Value Network 기본 2-step lookahead 적용
- gradient clipping, PPO rollout 묶음 학습, return/advantage 정규화 적용
- Rule-based 상대 최고 승률 체크포인트 저장
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

### Strategic Rule-based Agent

기존 Rule-based보다 윷놀이 전략을 더 많이 반영한 개선 기준선입니다.

각 행동을 복사한 환경에서 실제로 한 번 시뮬레이션한 뒤, 완주, 잡기, 진행도, 업힌 말 이동, 지름길 진입, 상대에게 잡힐 위험을 함께 점수화합니다. 특히 상대 말이 도/개/걸/윷/모로 내 말을 잡을 수 있는 확률을 계산해서 위험한 위치로 가는 수에는 벌점을 줍니다.

현재 기본 설정은 1,000게임 비교에서 55%를 넘긴 조합입니다.

- `finish_weight=130`
- `capture_weight=55`
- `progress_weight=3`
- `stack_weight=7`
- `danger_weight=28`
- `shortcut_weight=8`
- `counterplay_weight=18`

핵심은 `counterplay_weight`입니다. 내 수만 좋게 보는 것이 아니라, 내가 둔 뒤 상대가 바로 잡거나 완주할 수 있는 최선 반격을 계산해서 벌점을 줍니다.

실행 이름은 `strategic`입니다.

### Tabular Q-learning Agent

PyTorch 없이 Python 기본 기능만으로 실행되는 첫 강화학습 모델입니다.

상태별 Q값을 딕셔너리에 저장하고, 가능한 행동 중 Q값이 높은 행동을 선택합니다. 현재 행동은 단순히 말 4개 중 하나가 아니라 `윷 결과 5종 x 말 4개` 조합입니다.

### DQN Agent

one-hot 상태를 입력으로 받고, 최대 20개 행동에 대한 Q값을 출력합니다. 현재 보유한 윷 결과와 움직일 말을 함께 고르는 방식입니다.

이 모델은 `이 상태에서 이 행동을 하면 앞으로 얻을 보상이 얼마나 큰가`를 직접 학습합니다. 행동 수가 명확한 문제에서 쓰기 쉽고, 구현도 비교적 단순합니다.

다만 윷놀이는 상대가 어떤 수를 두는지, 잡기와 추가 던지기가 어떻게 이어지는지에 따라 미래가 크게 바뀝니다. 그래서 기본 DQN만으로는 Q값이 불안정하거나 과하게 낙관적으로 학습될 수 있습니다. 현재 코드는 이 문제를 줄이기 위해 상대 턴이 끝나고 내 차례가 돌아온 상태를 다음 상태로 저장하도록 수정되어 있습니다.

### Double DQN Agent

DQN에서 다음 행동 선택과 target Q값 평가를 분리한 모델입니다. 기본 DQN의 Q값 과대평가 문제를 줄이는 목적입니다.

기본 DQN은 `다음 상태에서 가장 좋아 보이는 행동`의 값을 그대로 target으로 쓰기 때문에 실제보다 Q값이 높게 잡히는 경우가 많습니다. Double DQN은 다음 행동은 현재 네트워크로 고르고, 그 행동의 값은 target network로 평가합니다.

이 프로젝트에서는 기본 DQN보다 안정적인 Q-learning 비교 모델로 쓰는 것이 좋습니다. Rule-based 상대를 이기고 싶은 목적이면 `dqn`보다 `double-dqn`을 먼저 돌려보는 편이 낫습니다.

### Dueling DQN Agent

상태 가치와 행동 이점을 분리해서 학습하는 DQN 변형입니다. 같은 상태에서 행동별 차이를 더 안정적으로 볼 수 있게 합니다.

네트워크 내부에서 `이 상태 자체가 좋은가`와 `이 행동이 다른 행동보다 얼마나 좋은가`를 나누어 계산합니다. 윷놀이처럼 어떤 상태는 이미 유리하거나 불리하고, 그 안에서 세부 행동 차이를 골라야 하는 게임에 잘 맞을 수 있습니다.

현재 구현은 Dueling 구조와 Double DQN target 방식을 같이 사용합니다. 그래서 이름은 `dueling-dqn`이지만 실제로는 더 안정적인 Dueling Double DQN에 가깝습니다.

### REINFORCE Agent

정책 확률 자체를 학습하는 가장 기본적인 policy gradient 모델입니다. 구현은 단순하지만 분산이 커서 학습이 불안정할 수 있습니다.

Q값을 따로 학습하는 대신, 상태를 보고 각 행동을 선택할 확률을 바로 학습합니다. 이긴 게임에서 했던 행동의 확률은 높이고, 진 게임에서 했던 행동의 확률은 낮추는 방향입니다.

장점은 구조가 단순하고 정책 기반 모델의 출발점으로 보기 좋다는 것입니다. 단점은 한 게임 결과 전체에 크게 흔들려서 학습 분산이 큽니다. 그래서 현재 코드에서는 return 정규화와 entropy 보너스를 넣어 너무 빨리 한 행동으로 굳어지는 것을 줄였습니다.

### A2C Agent

Actor-Critic 구조입니다. 정책을 담당하는 Actor와 상태 가치를 보는 Critic을 함께 학습합니다.

Actor는 `어떤 행동을 할지`를 고르고, Critic은 `현재 상태가 얼마나 좋은지`를 평가합니다. REINFORCE가 게임 결과만 보고 정책을 고치는 것보다, Critic이 기준선을 제공하기 때문에 일반적으로 더 안정적입니다.

현재 구현에서는 REINFORCE와 같은 policy network를 쓰되 value head를 함께 학습합니다. PPO보다 단순하고 빠르게 비교할 수 있는 Actor-Critic 기준 모델입니다.

### PPO Agent

정책이 한 번에 너무 크게 바뀌지 않도록 clipped objective를 사용하는 policy gradient 계열 모델입니다.

PPO는 Actor-Critic 계열 중 실험에서 자주 쓰이는 안정적인 방법입니다. 정책을 업데이트할 때 `이전 정책과 너무 달라지지 않는 범위` 안에서만 크게 보상을 주도록 clipping을 사용합니다.

현재 코드에서는 여러 게임을 모은 뒤(`--rollout-episodes`) 여러 번 업데이트(`--ppo-epochs`)합니다. REINFORCE나 A2C보다 튜닝할 값은 조금 많지만, 충분히 에피소드를 늘리면 policy gradient 계열 중 가장 기대해볼 만한 모델입니다.

### Value Network Agent

논문 방식에 더 가까운 기본 모델입니다.

가능한 행동을 하나씩 미리 시뮬레이션한 뒤, 각 행동으로 만들어지는 다음 상태를 신경망이 평가합니다. 그리고 다음 상태 가치가 가장 높은 행동을 선택합니다.

현재 기본 lookahead 깊이는 2입니다.

학습 중에는 현재 모델 혼자만 상대하지 않고, Random, Rule-based, 과거에 저장한 자기 자신의 스냅샷 모델을 섞어서 상대합니다. 이렇게 하면 한 가지 상대에게만 과적합되는 문제를 줄일 수 있습니다.

### Strategic Value Network Agent

Value Network의 예측 점수와 Strategic Rule-based의 전략 점수를 섞어서 행동을 고르는 개선 모델입니다.

학습 초반이나 데이터가 부족한 상황에서는 신경망 가치 예측이 흔들릴 수 있습니다. 이 모델은 같은 Value Network 구조를 쓰되, 행동 선택 단계에서 잡기, 완주, 위험 회피, 지름길 같은 전략 점수를 함께 반영합니다.

실행 이름은 `strategic-value`입니다. `--heuristic-weight` 값이 클수록 전략 점수의 영향이 커지고, 작을수록 신경망 예측을 더 많이 따릅니다.

### MCTS + Value Network Agent

학습된 Value Network를 평가 함수로 사용해서 가능한 행동을 여러 번 시뮬레이션합니다. 현재 구현은 가벼운 shallow search 형태입니다.

Value Network가 한 번의 다음 상태만 보고 선택한다면, MCTS + Value는 각 행동 뒤에 짧은 무작위 진행을 여러 번 붙여서 평균적으로 더 좋아 보이는 행동을 고릅니다.

학습 모델 자체를 크게 바꾸는 방식이라기보다는, 이미 학습된 Value Network를 더 잘 활용하는 선택 방식입니다. `--mcts-simulations`를 키우면 더 꼼꼼하게 보지만 실행 속도는 느려집니다.

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

## 추가 에이전트 실행

```bash
.venv/bin/python -m yut_rl.train --agent double-dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent dueling-dqn --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent strategic --eval-games 200
.venv/bin/python -m yut_rl.train --agent strategic-value --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent reinforce --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent a2c --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent ppo --episodes 1000 --eval-games 200
.venv/bin/python -m yut_rl.train --agent mcts-value --episodes 1000 --eval-games 200
```

## 기존 에이전트와 개선 에이전트 비교

아래 명령은 기존 Rule-based와 Strategic Rule-based, 기존 Value Network와 Strategic Value Network를 앞/뒤 순서를 바꿔가며 비교합니다.

```bash
.venv/bin/python -m yut_rl.compare --games 100
```

이미 학습된 Value 체크포인트를 같은 가중치로 일반 Value와 Strategic Value에 둘 다 적용해서 비교하려면:

```bash
.venv/bin/python -m yut_rl.compare --games 100 --value-model checkpoints/value.pt
```

출력 지표:

- `win_rate`: 승률
- `avg_turns`: 평균 턴 수
- `avg_captures`: 게임당 평균 잡기 횟수
- `avg_finished`: 게임 종료 시 평균 완주 말 수
- `avg_reward`: 평균 보상값

55% 이상 승률 조합을 자동 탐색하려면:

```bash
.venv/bin/python -m yut_rl.tune --games 100 --confirm-games 1000 --top-k 3 --value-model checkpoints/value.pt
```

현재 확인된 1,000게임 결과:

- `strategic_rule_based`: 57.4%
- `strategic_value`: 72.5%

## 에이전트별 성능을 올리는 추천 설정

### Value Network

현재 프로젝트에서 논문 방식에 가장 가까운 기본 모델입니다. Rule-based 상대 승률을 올리고 싶으면 Rule-based를 더 자주 만나게 하고, 과거 스냅샷도 유지하는 쪽이 좋습니다.

```bash
.venv/bin/python -m yut_rl.train \
  --agent value \
  --episodes 50000 \
  --eval-games 300 \
  --eval-interval 2500 \
  --rule-opponent-weight 6.0 \
  --random-opponent-weight 0.25 \
  --snapshot-opponent-weight 1.5 \
  --lookahead-depth 2 \
  --best-model checkpoints/best_value_vs_rule.pt
```

### DQN 계열

기본 DQN보다 `double-dqn`이나 `dueling-dqn`부터 비교하는 것을 추천합니다. Double DQN은 Q값 과대평가를 줄이고, Dueling DQN은 상태 가치와 행동별 장점을 나누어 봅니다.

```bash
.venv/bin/python -m yut_rl.train --agent double-dqn --episodes 20000 --target-sync 100
.venv/bin/python -m yut_rl.train --agent dueling-dqn --episodes 20000 --target-sync 100
```

### PPO / A2C / REINFORCE

Policy gradient 계열은 한 게임만 보고 바로 업데이트하면 흔들릴 수 있습니다. `--rollout-episodes`를 늘리면 여러 게임을 모아서 업데이트합니다.

```bash
.venv/bin/python -m yut_rl.train \
  --agent ppo \
  --episodes 20000 \
  --rollout-episodes 8 \
  --ppo-epochs 4 \
  --entropy-coef 0.01
```

### MCTS + Value Network

학습된 Value Network를 기반으로 시뮬레이션을 더 많이 하면 더 깊게 비교할 수 있지만 실행 속도는 느려집니다.

```bash
.venv/bin/python -m yut_rl.train \
  --agent mcts-value \
  --episodes 50000 \
  --mcts-simulations 64 \
  --mcts-rollout-depth 8
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
  --rule-opponent-weight 4.0 \
  --lookahead-depth 2
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

- `--agent`: 사용할 모델, `tabular`, `strategic`, `value`, `strategic-value`, `dqn`, `double-dqn`, `dueling-dqn`, `reinforce`, `a2c`, `ppo`, `mcts-value`
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
- `--lookahead-depth`: Value Network가 몇 단계까지 다음 행동을 볼지
- `--best-model`: Rule-based 상대 최고 승률 체크포인트 저장 경로
- `--reward-mode`: `dense`, `terminal`, `hybrid` 중 선택. 기본값은 `hybrid`
- `--dense-reward-scale`: `hybrid`에서 잡기/도착 같은 중간 보상을 얼마나 반영할지
- `--grad-clip`: gradient 폭주를 막기 위한 clipping 값
- `--heuristic-weight`: Strategic Value Network에서 전략 점수를 얼마나 섞을지
- `--target-sync`: target network를 몇 에피소드마다 동기화할지
- `--rollout-episodes`: PPO/A2C/REINFORCE에서 업데이트 전에 모을 게임 수
- `--ppo-epochs`: PPO 업데이트 반복 횟수
- `--ppo-clip`: PPO clipping 범위
- `--mcts-simulations`: MCTS + Value 평가 시 행동당 시뮬레이션 수
- `--mcts-rollout-depth`: MCTS에서 무작위 rollout을 몇 단계까지 볼지

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
