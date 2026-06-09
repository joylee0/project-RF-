from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common_env_experiment_utils import RESULT_DIR
from yut_rl.config_runner import make_opponent
from yut_rl.env import BOARD_POSITIONS, FINISH, HOME, START, YUT_OUTCOMES, YutEnv, advance


def check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    env = YutEnv(
        seed=0,
        state_encoder="raw",
        reward_function="minimal_dense",
        action_encoding="step",
        enable_action_mask=True,
    )
    obs = env.reset(seed=0)

    prob_sum = sum(prob for _, _, prob, _ in YUT_OUTCOMES)
    results.append(check("board size", len(BOARD_POSITIONS) == 29 and HOME in BOARD_POSITIONS, f"board_positions={len(BOARD_POSITIONS)}, home={HOME}"))
    results.append(check("yut probability", abs(prob_sum - 1.0) < 1e-9, f"probability_sum={prob_sum:.6f}, outcomes={YUT_OUTCOMES}"))
    results.append(check("action space", env.action_dim == 20, f"action_dim={env.action_dim}"))

    legal = env.legal_actions()
    mask = env.action_mask()
    mask_actions = [idx for idx, value in enumerate(mask) if value]
    results.append(check("legal action mask", sorted(legal) == mask_actions and len(mask) == env.action_dim, f"legal={legal}, mask_on={mask_actions}"))

    state_dims = {}
    for encoder in ["raw", "board", "engineered", "risk_aware"]:
        probe = YutEnv(seed=0, state_encoder=encoder, reward_function="sparse", action_encoding="step")
        state_dims[encoder] = len(probe.reset(seed=0))
    results.append(check("state dimension", all(dim > 0 for dim in state_dims.values()), json.dumps(state_dims, ensure_ascii=False)))

    env.pending_steps = [env.decode_action(legal[0])[1]]
    step_result = env.step(legal[0])
    results.append(check("reward output", isinstance(step_result.reward, float), f"reward={step_result.reward}, done={step_result.done}"))

    terminal_env = YutEnv(seed=1, state_encoder="raw", reward_function="sparse", action_encoding="step")
    terminal_env.reset(seed=1)
    terminal_env.positions[0] = [HOME, FINISH, FINISH, FINISH]
    terminal_env.positions[1] = [START, START, START, START]
    terminal_env.current_player = 0
    terminal_env.pending_steps = [1]
    terminal_result = terminal_env.step(terminal_env.encode_action(0, 1))
    results.append(
        check(
            "terminal condition",
            terminal_result.done and terminal_env.winner() == 0,
            f"positions={terminal_env.positions[0]}, reward={terminal_result.reward}",
        )
    )

    home_exact = advance(18, 1)
    home_pass = advance(HOME, 1)
    results.append(check("home rule", home_exact == HOME and home_pass == FINISH, f"advance(18,1)={home_exact}, advance(HOME,1)={home_pass}"))

    opponent_env = YutEnv(seed=2, state_encoder="raw", reward_function="sparse", action_encoding="step")
    opponent_env.reset(seed=2)
    opponent = make_opponent({"opponent_type": "rule_based"}, seed=2)
    opponent_action = opponent.act(opponent_env)
    results.append(check("opponent behavior", opponent_action in opponent_env.legal_actions(), f"opponent_action={opponent_action}"))

    a = YutEnv(seed=42, state_encoder="raw", reward_function="sparse", action_encoding="step")
    b = YutEnv(seed=42, state_encoder="raw", reward_function="sparse", action_encoding="step")
    obs_a = a.reset(seed=42)
    obs_b = b.reset(seed=42)
    reproducible = obs_a == obs_b and a.legal_actions() == b.legal_actions() and a.pending_steps == b.pending_steps
    results.append(check("seed reproducibility", reproducible, f"pending_a={a.pending_steps}, pending_b={b.pending_steps}"))

    path = RESULT_DIR / "common_env_validation.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status}: {item['check']} - {item['detail']}")
    if not all(item["passed"] for item in results):
        raise SystemExit(1)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
