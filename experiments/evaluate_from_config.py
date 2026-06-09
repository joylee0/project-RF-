from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yut_rl.config_runner import evaluate_config_agent, load_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    if args.model is None:
        candidate = Path("results/config_runs") / Path(args.config).stem / "model.pt"
        model_path = candidate if candidate.exists() else None
    else:
        model_path = Path(args.model)
    result = evaluate_config_agent(config, model_path=model_path, games=args.games)
    result["model_path"] = "" if model_path is None else str(model_path)
    result["config"] = args.config

    out_path = Path(args.out or f"results/config_eval/{Path(args.config).stem}_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
