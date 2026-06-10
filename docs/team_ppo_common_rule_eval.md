# TeamPPO vs Common Rule-Based 참고 평가

이 문서는 팀원 PPO가 `common_rule_based`를 상대로 어느 정도 성능을 내는지 확인한 참고 평가입니다.

주의: 이 결과는 **MyAgent와 TeamPPO의 직접 대결이 아닙니다.**  
직접 대결 결과는 [TeamPPO vs MyAgent 공통 Paired Evaluation 보고서](team_ppo_vs_my_agent_common_eval.md)를 기준으로 봐야 합니다.

## 평가 조건

- protocol: `common_rule_based_paired_v1`
- opponent: `common_rule_based`
- observation mode: `tactical`
- training reward mode: `terminal`
- base seeds: `100000~102499`
- paired seeds: 2,500
- games per model: 5,000
- deterministic policy: true

## 결과

| training seed | wins | losses | win rate | 선공 | 후공 | passed | illegal | errors |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 0 | 2921 | 2079 | 58.42% | 59.00% | 57.84% | false | 0 | 0 |
| 1 | 3023 | 1977 | 60.46% | 60.76% | 60.16% | true | 0 | 0 |
| 2 | 3020 | 1980 | 60.40% | 61.32% | 59.48% | true | 0 | 0 |

## 종합

```text
mean win rate: 59.76%
pooled win rate: 59.76%
seed std: 0.95%p
passed seeds: 2 / 3
pooled games: 15,000
illegal actions: 0
evaluation errors: 0
```

## 해석

이 평가는 팀원 PPO가 공통 rule-based baseline을 상대로 60% 학습 목표에 얼마나 근접했는지 보여주는 참고 자료입니다.  
3-seed 평균 승률은 59.76%로 목표에 거의 도달했습니다. 프로젝트의 최종 직접 비교는 `TeamPPO vs MyAgent(project_rf_rule)` 평가로 분리합니다.
