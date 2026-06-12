# 공통 평가 프로토콜

## 목적

공통 paired evaluation protocol은 서로 다른 프로젝트에서 학습한 agent를 같은 기준으로 비교하기 위해 도입했다.

내부 tournament 결과는 각 프로젝트의 개발 과정에서는 의미가 있지만, 환경과 상대 agent가 다르면 직접 비교하기 어렵다. 따라서 최종 비교에서는 공통 윷놀이 규칙, 동일한 seed 조건, 선공/후공 교대 방식을 사용했다.

## 공통 환경

최종 평가는 다음 윷놀이 규칙을 기준으로 한다.

- 2인 게임
- 플레이어당 말 4개
- 잡기와 업기 사용
- 분기점에 정확히 도착하면 다음 이동부터 자동 지름길 진입
- 윷/모 추가 턴 사용
- 뒷도 없음
- 낙 없음
- 후진 이동 없음
- agent가 직접 지름길을 선택하지 않음
- action space: `4 pieces x 5 yut results = 20 actions`
- illegal action은 action mask로 제거

윷 결과 확률은 다음과 같다.

| 결과 | 이동량 | 확률 |
| --- | ---: | ---: |
| 도 | 1 | 0.1536 |
| 개 | 2 | 0.3456 |
| 걸 | 3 | 0.3456 |
| 윷 | 4 | 0.1296 |
| 모 | 5 | 0.0256 |

## Paired Evaluation 방식

각 base seed마다 두 게임을 실행한다.

| 게임 | 선공 | 후공 |
| --- | --- | --- |
| Game A | Agent 1 | Agent 2 |
| Game B | Agent 2 | Agent 1 |

기본 평가 설정은 다음과 같다.

- base seed: 2,500개
- seed당 2게임
- 총 5,000판
- deterministic policy 사용
- 같은 윷 확률 사용
- 미래 윷 결과 참조 금지
- illegal action 수 기록
- evaluation error 수 기록

이 방식은 같은 seed 조건에서 두 agent가 모두 선공과 후공을 경험하게 하므로, 선공 편향을 줄이는 데 도움이 된다.

## 평가 지표

평가에서는 다음 지표를 기록한다.

- 전체 게임 수
- paired seed 수
- 전체 승률
- 선공 승률
- 후공 승률
- 평균 턴 수
- 평균 잡기 수
- 평균 완주 말 수
- illegal action 수
- evaluation error 수
- 필요 시 confidence interval

## 내부 결과와 최종 결과의 분리

내부 tournament, 학습 중간 평가, smoke test는 개발 과정의 참고 자료다.  
최종 프로젝트 간 비교는 common paired evaluation 결과를 기준으로 해석한다.

## 주요 결과

### Common Rule-based Evaluation

| Agent | Win Rate |
| --- | ---: |
| RL-project | 59.76% |
| project-RF | 59.46% |

### Head-to-Head Evaluation

| Metric | Value |
| --- | ---: |
| project-RF Hybrid win rate | 53.98% |
| 95% Wilson CI | 53.18% - 54.78% |

### Ablation

| 설정 | Win Rate |
| --- | ---: |
| project-RF tactical prior 제거 | 17.87% |
| RL-project network-only 기준 | 82.13% |

## 재현 명령어

공통 환경 검증:

```bash
python experiments/validate_common_env.py
```

로컬 common paired evaluation 실행:

```bash
python experiments/common_paired_evaluation.py \
  --my-agent ppo_capture_imitation \
  --friend-agent friend_ppo \
  --num-paired-seeds 2500 \
  --total-games 5000 \
  --seed 42 \
  --output-dir results/common_paired_eval
```

Team PPO와 project-RF agent 직접 평가:

```bash
python experiments/team_ppo_vs_my_agent_common_eval.py \
  --team-repo /path/to/RL-yutnori \
  --model-path /path/to/model.zip \
  --training-seed 1 \
  --num-paired-seeds 2500 \
  --seed-start 100000 \
  --output-dir results/team_ppo_vs_my_agent_common_eval/seed1
```
