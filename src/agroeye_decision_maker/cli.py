from __future__ import annotations

import argparse
import json
from pathlib import Path

from agroeye_decision_maker.pipeline import run_training
from agroeye_decision_maker.runtime import DecisionRuntime
from agroeye_decision_maker.utils.config import load_yaml, seed_everything


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    safety_cfg = load_yaml(args.safety)
    seed_everything(int(cfg["project"]["seed"]))
    artifacts = run_training(cfg, safety_cfg)
    print(json.dumps(artifacts.metrics, indent=2))


def cmd_decide(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    safety_cfg = load_yaml(args.safety)
    runtime = DecisionRuntime(cfg, safety_cfg)
    req = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    resp = runtime.decide(req.get("timestamp_utc"), req.get("sensors", {}), req.get("override_config"))
    print(json.dumps(resp, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AgroEye Decision Maker CLI")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--safety", default="configs/safety.yaml")

    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train/evaluate and produce artifacts/report")
    train_p.set_defaults(func=cmd_train)

    decide_p = sub.add_parser("decide", help="Run one decision from input JSON")
    decide_p.add_argument("--input", required=True, help="Path to JSON request")
    decide_p.set_defaults(func=cmd_decide)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
