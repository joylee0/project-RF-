from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common_env_experiment_utils import BASE_CONFIG, RESULT_DIR, SEEDS, plot_learning_curve, write_final_summary, write_rows
from yut_rl.config_runner import run_learning_curve_experiment


REWARD_CONFIGS = [
    ("SparseReward", {"reward_function": "sparse"}),
    ("MinimalDenseReward", {"reward_function": "minimal_dense"}),
    ("BalancedTacticalReward", {"reward_function": "balanced_tactical"}),
    ("RiskAwareReward", {"reward_function": "risk_aware"}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward function ablation on the common Yut env.")
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--state", choices=["raw", "board"], default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for config_name, overrides in REWARD_CONFIGS:
        for seed in args.seeds:
            config = {
                **BASE_CONFIG,
                "algorithm": "ppo",
                "state_encoder": args.state,
                **overrides,
            }
            print(f"running reward ablation: {config_name}, seed={seed}")
            rows.extend(
                run_learning_curve_experiment(
                    config,
                    config_name=config_name,
                    seed=seed,
                    train_episodes=args.train_episodes,
                    eval_interval=args.eval_interval,
                    eval_games=args.eval_games,
                )
            )

    csv_path = RESULT_DIR / "reward_ablation.csv"
    write_rows(csv_path, rows)
    plot_learning_curve(rows, RESULT_DIR / "reward_ablation_learning_curve.png", "Reward Design Ablation")
    write_final_summary(RESULT_DIR)
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
