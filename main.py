from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running without installing the package:
#   python main.py --config configs/loveda.yaml
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from landcover.config import deep_update, load_config
from landcover.training import run_experiment


def parse_overrides(items: list[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE overrides.

    Supports nested keys via dots: ebct.use=true
    Values: try json.loads; fall back to string.
    """
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Override must be KEY=VALUE, got: {it}")
        k, v = it.split("=", 1)
        try:
            v_parsed = json.loads(v)
        except Exception:
            v_parsed = v
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v_parsed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/loveda.yaml")
    ap.add_argument("--tag", default="run", help="Run tag used in checkpoint filenames")
    ap.add_argument("--override", nargs="*", default=[], help="Config overrides, e.g. lr=0.0001 ebct.use=true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.override:
        deep_update(cfg, parse_overrides(args.override))

    run_experiment(cfg, tag=args.tag)


if __name__ == "__main__":
    main()
