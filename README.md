# 윷놀이 강화학습 공통 평가 프로젝트

한국 전통 놀이 윷놀이를 강화학습 환경으로 구현하고, 고정된 Rule-based Agent를 상대하는 PPO 기반 강화학습 모델을 학습·평가하는 프로젝트입니다.

최종 평가는 **common env paired evaluation** 기준으로 수행했습니다.  
이는 같은 환경과 같은 base seed에서 선공/후공을 한 번씩 바꿔 실행하는 방식입니다. 이 방식으로 선공 편향을 줄이고, 동일한 윷 확률과 action mask 조건에서 PPO가 고정 Rule-based Agent를 상대로 어느 정도 성능을 내는지 계산합니다.

## 핵심 결론

팀원이 학습한 `ppo_common_rule_40m_subproc` 모델과 내가 설계한 Rule-based Agent를 공통 env에서 직접 평가했습니다.

- MyAgent는 팀원 저장소에 포팅된 `project_rf_rule`로 평가했습니다.
- TeamPPO는 `David-Nam/RL-yutnori`의 `bests/ppo_common_rule_40m_subproc` 모델입니다.
- 각 PPO seed마다 2,500 paired seeds, 총 5,000판을 평가했습니다.

| TeamPPO seed | TeamPPO 승률 | MyAgent 승률 | illegal | errors |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 57.26% | 42.74% | 0 | 0 |
| 1 | 57.56% | 42.44% | 0 | 0 |
| 2 | 55.08% | 44.92% | 0 | 0 |

종합:

```text
TeamPPO 평균 승률: 56.63%
MyAgent 평균 승률: 43.37%
총 게임 수: 15,000
illegal actions: 0
evaluation errors: 0
```

팀원의 목표는 내가 만든 Rule-based Agent를 상대로 약 60% 승률을 달성하는 것이었습니다.  
공통 `common_rule_based` 기준 참고 평가는 평균 59.76%로 목표에 거의 도달했고, MyAgent(`project_rf_rule`)와의 직접 paired evaluation에서는 평균 56.63%를 기록했습니다.

자세한 보고서:

- [최종 결과 보고서](result.md)
- [PPT 구성안](docs/ppt_report_structure.md)
- [TeamPPO vs MyAgent 직접 평가 보고서](docs/team_ppo_vs_my_agent_common_eval.md)
- [TeamPPO vs common rule-based 참고 평가](docs/team_ppo_common_rule_eval.md)

## 공통 규칙

- 플레이어 수: 2명
- 플레이어당 말: 4개
- 뒷도, 낙, 후진 이동 없음
- 지름길 선택은 action에 포함하지 않음
- 분기점에 정확히 도착하면 다음 이동부터 자동 지름길 진입
- 잡기와 업기는 도착 칸에서만 발생
- HOME에 정확히 도착하면 즉시 완주하지 않고 보드 위에 남음
- HOME을 통과하면 FINISHED
- HOME에 있는 상대 말도 잡을 수 있음
- 윷/모 20회 연속 즉시 승리 없음

윷 확률:

| 결과 | 이동량 | 확률 |
| --- | ---: | ---: |
| 도 | 1 | 0.1536 |
| 개 | 2 | 0.3456 |
| 걸 | 3 | 0.3456 |
| 윷 | 4 | 0.1296 |
| 모 | 5 | 0.0256 |

## 주요 파일

```text
common_rule_based_env.py
  공통 rule-based 평가 환경

experiments/common_paired_evaluation.py
  로컬 PPO/checkpoint끼리 common paired evaluation을 수행하는 스크립트

experiments/team_ppo_vs_my_agent_common_eval.py
  팀원 MaskablePPO model.zip과 MyAgent(project_rf_rule)를 직접 평가하는 스크립트

experiments/validate_common_env.py
  공통 env 규칙 검증 스크립트

experiments/ablation_state_design.py
experiments/ablation_reward_design.py
experiments/compare_rl_algorithms_common_env.py
  state/reward/algorithm ablation 실험 스크립트

yut_rl/
  기본 윷놀이 환경, agent, config 기반 학습/평가 코드

docs/
  GitHub용 최종 보고서와 PPT 구성안
```

## 보고서 구성

최종 보고서인 [result.md](result.md)는 다음 흐름으로 작성했습니다.

- 프로젝트 목표
- 윷놀이 강화학습 환경 설계
- state/action/reward 설계
- agent와 model 설명
- PPO 학습 및 개선 과정
- common env paired evaluation
- 최종 평가 결과
- 어려웠던 점과 해결방법
- 최종 결론

PPT 발표 자료는 [PPT 구성안](docs/ppt_report_structure.md)의 슬라이드 순서를 기준으로 작성할 수 있습니다.

## 설치

기본 프로젝트 실행:

```bash
python -m pip install -r requirements.txt
```

팀원 `model.zip`을 직접 로드하려면 `stable-baselines3`, `sb3-contrib`, `gymnasium`이 필요합니다. 현재 `requirements.txt`에 포함되어 있습니다.

## 공통 env 검증

```bash
python experiments/validate_common_env.py
```

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

## TeamPPO vs MyAgent 직접 평가

먼저 팀원 저장소를 내려받습니다.

```bash
git clone https://github.com/David-Nam/RL-yutnori.git /private/tmp/RL-yutnori-team-model
```

seed 1 모델 예시:

```bash
python experiments/team_ppo_vs_my_agent_common_eval.py \
  --team-repo /private/tmp/RL-yutnori-team-model \
  --model-path /private/tmp/RL-yutnori-team-model/bests/ppo_common_rule_40m_subproc/common_rule_based_seed_1_40m_nenv12_tactical/model.zip \
  --training-seed 1 \
  --num-paired-seeds 2500 \
  --seed-start 100000 \
  --output-dir results/team_ppo_vs_my_agent_common_eval/seed1
```

3개 seed 전체를 비교하려면 seed 0, 1, 2의 `model.zip` 경로를 바꿔 반복 실행합니다.

## 참고: TeamPPO vs common rule-based 평가

이 평가는 직접 대결이 아니라 팀원 PPO가 공통 rule-based agent를 상대로 어느 정도 성능을 내는지 확인하는 참고용입니다.

```bash
/private/tmp/sb3evalvenv/bin/python \
  /private/tmp/RL-yutnori-team-model/scripts/evaluate_common_rule.py \
  --model-path /private/tmp/RL-yutnori-team-model/bests/ppo_common_rule_40m_subproc/common_rule_based_seed_1_40m_nenv12_tactical/model.zip \
  --training-seed 1 \
  --device cpu \
  --seed-start 100000 \
  --seed-count 2500 \
  --no-progress-bar \
  --output results/team_common_rule_eval_seed1_5000.json
```

## 결과 파일 정책

`results/`와 `checkpoints/`는 `.gitignore`에 포함되어 있어 GitHub에 올리지 않습니다.  
최종 결과는 `docs/`의 Markdown 보고서에 정리합니다.

## 주의

최종 결과는 `TeamPPO vs MyAgent(project_rf_rule)` 직접 대결 결과를 기준으로 해석합니다.  
여기서 MyAgent는 고정 Rule-based Agent 역할을 하고, TeamPPO는 해당 agent를 상대하도록 학습된 PPO 모델입니다.
