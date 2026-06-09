# FriendPPO vs MyAgent 공통 Paired Evaluation 보고서

이 보고서는 친구 PPO와 내 agent를 같은 common env에서 직접 평가한 결과입니다.

MyAgent는 친구 저장소에 포팅된 `project_rf_rule`로 평가했습니다.  
FriendPPO는 `David-Nam/RL-yutnori`의 `bests/ppo_common_rule_40m_subproc`에 공개된 `model.zip`입니다.

## 평가 조건

- protocol: `project_rf_rule_paired_v1`
- env: 친구 저장소의 common Yutnori env
- opponent: `project_rf_rule`
- opponent interpretation: MyAgent ported in the friend repository
- observation mode: `tactical`
- FriendPPO policy: deterministic
- base seeds: `100000~102499`
- paired seeds: 2,500
- games per seed: 2
- games per model: 5,000
- total games across 3 PPO seeds: 15,000

## 평가 방식

각 base seed마다 두 게임을 실행했습니다.

```text
Game A: FriendPPO 선공, MyAgent 후공
Game B: MyAgent 선공, FriendPPO 후공
```

이렇게 선공/후공을 한 쌍으로 묶어 선공 편향을 완화했습니다.

## 결과

| FriendPPO seed | FriendPPO 승률 | MyAgent 승률 | Friend 선공 | Friend 후공 | MyAgent 선공 | MyAgent 후공 | illegal | errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 57.26% | 42.74% | 56.80% | 57.72% | 42.28% | 43.20% | 0 | 0 |
| 1 | 57.56% | 42.44% | 58.96% | 56.16% | 43.84% | 41.04% | 0 | 0 |
| 2 | 55.08% | 44.92% | 56.36% | 53.80% | 46.20% | 43.64% | 0 | 0 |

## 종합

```text
FriendPPO 평균 승률: 56.63%
MyAgent 평균 승률: 43.37%
FriendPPO pooled 승률: 56.63%
MyAgent pooled 승률: 43.37%
FriendPPO seed std: 1.11%p
MyAgent seed std: 1.11%p
총 게임 수: 15,000
illegal actions: 0
evaluation errors: 0
```

## 해석

공통 env와 paired seed 조건에서 FriendPPO가 MyAgent를 직접 상대했을 때, 세 개 학습 seed 모두에서 FriendPPO가 우세했습니다.

이 결과는 다음 두 결과와 분리해서 해석해야 합니다.

- 기존 tournament 결과
- FriendPPO vs `common_rule_based` 참고 평가

최종 직접 비교 기준은 이 문서의 `FriendPPO vs MyAgent(project_rf_rule)` 결과입니다.

## 한계

- MyAgent는 친구 저장소에 포팅된 `project_rf_rule` 구현 기준입니다.
- 두 프로젝트의 원본 env가 완전히 같지 않기 때문에, 직접 비교는 common env에 올라간 구현을 기준으로 합니다.
- 5,000판 평가도 stochastic game 특성상 seed 영향이 남을 수 있습니다.

## 재현 명령어

```bash
python experiments/friend_ppo_vs_my_agent_common_eval.py \
  --friend-repo /private/tmp/RL-yutnori-friend-model \
  --model-path /private/tmp/RL-yutnori-friend-model/bests/ppo_common_rule_40m_subproc/common_rule_based_seed_1_40m_nenv12_tactical/model.zip \
  --training-seed 1 \
  --num-paired-seeds 2500 \
  --seed-start 100000 \
  --output-dir results/friend_ppo_vs_my_agent_common_eval/seed1
```
