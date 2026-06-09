from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from yut_rl.config_runner import evaluate_config_agent, train_from_config_dict


BASE = {
    "action_encoding": "step",
    "action_masking": True,
    "opponent_type": "rule_based",
    "algorithm": "dqn",
    "seed": 7,
    "train_episodes": 200,
    "eval_games": 100,
}

STATE_EXPERIMENTS = [
    ("raw_sparse", {"state_encoder": "raw", "reward_function": "sparse"}),
    ("board_sparse", {"state_encoder": "board", "reward_function": "sparse", "action_encoding": "piece_yut"}),
    ("engineered_sparse", {"state_encoder": "engineered", "reward_function": "sparse"}),
    ("engineered_balanced", {"state_encoder": "engineered", "reward_function": "balanced_tactical"}),
    ("risk_aware", {"state_encoder": "risk_aware", "reward_function": "risk_aware"}),
]

REWARD_EXPERIMENTS = [
    ("raw_sparse", {"state_encoder": "raw", "reward_function": "sparse"}),
    ("raw_minimal_dense", {"state_encoder": "raw", "reward_function": "minimal_dense"}),
    ("raw_balanced", {"state_encoder": "raw", "reward_function": "balanced_tactical"}),
    ("raw_capture_heavy", {"state_encoder": "raw", "reward_function": "capture_heavy"}),
    ("raw_risk_aware", {"state_encoder": "raw", "reward_function": "risk_aware"}),
]


def run_experiments(items: list[tuple[str, dict]], group: str, out_dir: Path) -> pd.DataFrame:
    rows = []
    for name, overrides in items:
        config = {**BASE, **overrides}
        run_dir = out_dir / "runs" / group / name
        summary = train_from_config_dict(config, run_dir)
        result = evaluate_config_agent(config, model_path=summary["model_path"])
        curve = summary["learning_curve"]
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        rows.append(
            {
                "group": group,
                "experiment": name,
                **config,
                **{k: v for k, v in result.items() if k != "learning_curve"},
                "model_path": summary["model_path"],
                "learning_curve_path": str(run_dir / "learning_curve.json"),
                "final_train_win_rate": summary["final_train_win_rate"],
            }
        )
        (run_dir / "learning_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x], df[y], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["avg_captures"], df["avg_finished_pieces"], s=80, color="#F58518")
    for _, row in df.iterrows():
        ax.annotate(row["experiment"], (row["avg_captures"], row["avg_finished_pieces"]), fontsize=8)
    ax.set_xlabel("avg_captures")
    ax.set_ylabel("avg_finished_pieces")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_curves(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in df.iterrows():
        curve = json.loads(Path(row["learning_curve_path"]).read_text(encoding="utf-8"))
        if not curve:
            continue
        episodes = [item["episode"] for item in curve]
        returns = [item["episode_return"] for item in curve]
        ax.plot(episodes, returns, label=row["experiment"], alpha=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("episode_return")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "results" / "state_reward_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_df = run_experiments(STATE_EXPERIMENTS, "state", out_dir)
    reward_df = run_experiments(REWARD_EXPERIMENTS, "reward", out_dir)
    config_df = pd.concat([state_df, reward_df], ignore_index=True)

    state_df.to_csv(out_dir / "state_ablation.csv", index=False)
    reward_df.to_csv(out_dir / "reward_ablation.csv", index=False)
    config_df.to_csv(out_dir / "config_summary.csv", index=False)

    plot_bar(state_df, "experiment", "win_rate", "State encoder win rate", out_dir / "state_encoder_win_rate.png")
    plot_bar(reward_df, "experiment", "win_rate", "Reward function win rate", out_dir / "reward_function_win_rate.png")
    plot_learning_curves(state_df, out_dir / "learning_curve_by_state.png")
    plot_learning_curves(reward_df, out_dir / "learning_curve_by_reward.png")
    plot_bar(reward_df, "experiment", "danger_move_rate", "Danger move rate by reward", out_dir / "danger_move_rate_by_reward.png")
    plot_tradeoff(reward_df, out_dir / "capture_vs_finish_tradeoff.png")

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
