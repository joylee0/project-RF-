from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yut_rl.config_runner import load_simple_yaml, train_from_config_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    config_name = Path(args.config).stem
    out_dir = Path(args.out_dir or f"results/config_runs/{config_name}")
    summary = train_from_config_dict(config, out_dir)
    print(f"saved model: {summary['model_path']}")
    print(f"final_train_win_rate: {summary['final_train_win_rate']:.3f}")


if __name__ == "__main__":
    main()
