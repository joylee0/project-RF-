from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "results" / "common_env"
MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
SEEDS = [0, 42, 2025]

BASE_CONFIG = {
    "action_encoding": "step",
    "action_masking": True,
    "opponent_type": "rule_based",
    "train_episodes": 200,
    "eval_interval": 25,
    "eval_games": 50,
}

METRIC_FIELDS = [
    "train_episode",
    "average_episode_return",
    "moving_average_return",
    "evaluation_win_rate",
    "moving_average_win_rate",
    "avg_turns",
    "avg_captures",
    "avg_finished_pieces",
    "invalid_action_rate",
    "danger_move_rate",
    "capture_success_rate",
    "seed",
    "config_name",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_learning_curve(rows: list[dict], path: Path, title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipped plot {path}: matplotlib unavailable ({exc})")
        return

    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row["config_name"], {}).setdefault(int(row["train_episode"]), []).append(float(row["evaluation_win_rate"]))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, episode_map in sorted(grouped.items()):
        episodes = sorted(episode_map)
        means = [mean(episode_map[episode]) for episode in episodes]
        ax.plot(episodes, means, marker="o", linewidth=1.6, label=name)
    ax.set_title(title)
    ax.set_xlabel("train_episode")
    ax.set_ylabel("evaluation_win_rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_final_rows(paths: list[Path]) -> list[dict]:
    summary = []
    for path in paths:
        rows = read_rows(path)
        if not rows:
            continue
        final_by_seed: dict[tuple[str, int], dict] = {}
        for row in rows:
            key = (row["config_name"], int(row["seed"]))
            if key not in final_by_seed or int(row["train_episode"]) > int(final_by_seed[key]["train_episode"]):
                final_by_seed[key] = row
        by_config: dict[str, list[dict]] = {}
        for row in final_by_seed.values():
            by_config.setdefault(row["config_name"], []).append(row)
        for config_name, config_rows in sorted(by_config.items()):
            wins = [float(row["evaluation_win_rate"]) for row in config_rows]
            returns = [float(row["moving_average_return"]) for row in config_rows]
            danger = [float(row["danger_move_rate"]) for row in config_rows]
            summary.append(
                {
                    "experiment_file": path.name,
                    "config_name": config_name,
                    "seeds": ",".join(str(row["seed"]) for row in sorted(config_rows, key=lambda item: int(item["seed"]))),
                    "final_win_rate_mean": mean(wins),
                    "final_win_rate_std": pstdev(wins) if len(wins) > 1 else 0.0,
                    "final_return_mean": mean(returns),
                    "final_return_std": pstdev(returns) if len(returns) > 1 else 0.0,
                    "danger_move_rate_mean": mean(danger),
                    "num_seeds": len(config_rows),
                }
            )
    return summary


def write_final_summary(out_dir: Path = RESULT_DIR) -> None:
    paths = [
        out_dir / "state_ablation.csv",
        out_dir / "reward_ablation.csv",
        out_dir / "algorithm_comparison.csv",
    ]
    summary = aggregate_final_rows(paths)
    csv_path = out_dir / "final_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_file",
        "config_name",
        "seeds",
        "final_win_rate_mean",
        "final_win_rate_std",
        "final_return_mean",
        "final_return_std",
        "danger_move_rate_mean",
        "num_seeds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    write_report(summary, ROOT / "report_common_env_appendix.md")


def write_report(summary: list[dict], path: Path) -> None:
    by_file: dict[str, list[dict]] = {}
    for row in summary:
        by_file.setdefault(row["experiment_file"], []).append(row)

    def best(file_name: str, metric: str, reverse: bool = True) -> dict | None:
        rows = by_file.get(file_name, [])
        if not rows:
            return None
        return sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)[0]

    best_state = best("state_ablation.csv", "final_win_rate_mean")
    best_reward = best("reward_ablation.csv", "final_win_rate_mean")
    sparse_reward = next((row for row in by_file.get("reward_ablation.csv", []) if row["config_name"] == "SparseReward"), None)
    best_algo = best("algorithm_comparison.csv", "final_win_rate_mean")
    unstable = sorted(summary, key=lambda row: float(row["final_win_rate_std"]), reverse=True)[:3]

    lines = [
        "# Common Env Appendix",
        "",
        "이 문서는 tournament 승률 경쟁과 분리된 common env 실험 요약입니다.",
        "해석 기준은 최종 overall win rate 하나가 아니라 learning curve, sample efficiency, seed 안정성입니다.",
        _coverage_note(summary),
        "",
        "## State Design",
        _describe_best(best_state, "state 설계"),
        "",
        "## Reward Design",
        _describe_reward(best_reward, sparse_reward),
        "",
        "## Algorithm Stability",
        _describe_best(best_algo, "알고리즘"),
        "",
        "## Seed Variance",
    ]
    if unstable:
        for row in unstable:
            lines.append(
                f"- {row['experiment_file']} / {row['config_name']}: "
                f"win_rate_std={float(row['final_win_rate_std']):.4f}, "
                f"return_std={float(row['final_return_std']):.4f}"
            )
    else:
        lines.append("- 아직 실행된 ablation CSV가 없어 seed 편차를 계산하지 못했습니다.")

    lines.extend(
        [
            "",
            "## Friend Env vs Our Env Difference Checklist",
            "- route table: HOME exact landing, HOME pass-through finish, shortcut auto-entry timing이 동일한지 확인해야 합니다.",
            "- probability: 도/개/걸/윷/모 확률이 0.1536/0.3456/0.3456/0.1296/0.0256으로 고정되어야 합니다.",
            "- action space: 지름길 선택 없이 piece x yut-result action만 비교해야 합니다.",
            "- reward: 평가에서는 모델별 별도 승리 조건이나 tactical prior를 섞지 않아야 합니다.",
            "- opponent: 같은 rule_based opponent와 같은 seed schedule을 사용해야 학습 곡선 비교가 공정합니다.",
            "",
            "## Applied Setting Separation",
            "- StrategicValueNetworkAgent는 strong heuristic baseline으로만 해석합니다.",
            "- CaptureAwarePPO, tactical prior, imitation learning은 순수 RL 비교가 아니라 applied setting으로 분리합니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _describe_best(row: dict | None, label: str) -> str:
    if row is None:
        return f"- 아직 {label} 실험 결과 CSV가 없어 결론을 생성하지 못했습니다."
    return (
        f"- 현재 실행 결과 기준 가장 유리한 {label}: {row['config_name']} "
        f"(mean_win_rate={float(row['final_win_rate_mean']):.4f}, "
        f"std={float(row['final_win_rate_std']):.4f})."
    )


def _coverage_note(summary: list[dict]) -> str:
    if not summary:
        return "주의: 아직 실행된 결과가 없어 해석은 생성되지 않았습니다."
    min_seeds = min(int(row["num_seeds"]) for row in summary)
    if min_seeds < 3:
        return "주의: 현재 요약은 smoke/test run 결과를 포함할 수 있습니다. 최종 해석은 seeds=[0,42,2025] 전체 실행 후 갱신하세요."
    return "모든 config가 3개 seed 기준으로 집계되었습니다."


def _describe_reward(best_reward: dict | None, sparse_reward: dict | None) -> str:
    if best_reward is None:
        return "- 아직 reward ablation 결과 CSV가 없어 sparse reward 대비 개선 여부를 계산하지 못했습니다."
    if sparse_reward is None:
        return _describe_best(best_reward, "reward 설계")
    improvement = float(best_reward["final_win_rate_mean"]) - float(sparse_reward["final_win_rate_mean"])
    return (
        f"- SparseReward 대비 가장 좋은 reward: {best_reward['config_name']} "
        f"(win_rate 개선 {improvement:+.4f})."
    )
