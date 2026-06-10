# State / Action / Reward 설계 위치 분석

| 파일 경로 | 함수/클래스명 | 현재 역할 | 수정 필요 여부 |
| --- | --- | --- | --- |
| `yut_rl/env.py` | `YutEnv.observe_for` | 기존 observation 생성 위치. 위치 one-hot, pending 윷 결과, current player 정보를 직접 구성했다. | 완료: `state_encoder`로 위임 |
| `agents/ppo_agent.py` | `build_state` | PPO 전용 engineered state 생성. can_capture, can_finish, danger, distance feature 포함. | 필요: 새 encoder와 점진 통합 |
| `yut_rl/env.py` | `encode_action`, `decode_action`, `YutEnv.legal_actions` | 기존 step-major action encoding과 legal action 생성 담당. | 완료: `action_encoding`으로 위임 |
| `yut_rl/env.py` | `YutEnv.step` | 이동, 잡기, 업기, 턴 전환, 기존 dense reward 계산을 함께 처리. | 완료: `reward_function`으로 분리 가능 |
| `yut_rl/train.py` | `shape_reward` | terminal/hybrid reward 후처리. 학습 루프 내부에 reward mode가 섞여 있음. | 필요: config runner에서는 reward function 사용 |
| `train/train_ppo.py` | `RewardProfile`, `shaped_reward` | PPO 실험 전용 reward shaping. capture/danger/shortcut feature 포함. | 필요: 새 reward class와 비교 대상으로 유지 |
| `yut_rl/agents.py` | `StrategicRuleBasedAgent.score_breakdown` | strong heuristic baseline과 tactical score 계산. | 유지: 최종 TeamPPO 학습 모델이 아닌 비교/분석용 baseline으로 사용 |
| `yut_rl/agents.py` | `DQNAgent`, `A2CAgent`, `PPOAgent` | 기존 신경망 agent 구현. env observation/action mask에 의존. | 부분 완료: config env와 연결 가능 |

## 요약

기존 프로젝트는 agent 성능 개선을 중심으로 발전하면서 state feature, reward shaping, action encoding이 여러 파일에 흩어져 있었다. 이번 수정으로 새 실험 경로에서는 다음처럼 분리된다.

- `yut_rl/state_encoders.py`: Raw / Board / Engineered / RiskAware state representation
- `yut_rl/reward_functions.py`: Sparse / MinimalDense / BalancedTactical / CaptureHeavy / RiskAware reward
- `yut_rl/action_encodings.py`: StepActionEncoding / PieceYutActionEncoding
- `train/train_from_config.py`: config 기반 학습
- `experiments/evaluate_from_config.py`: config 기반 평가
