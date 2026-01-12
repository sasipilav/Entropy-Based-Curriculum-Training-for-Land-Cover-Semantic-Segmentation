from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landcover.config import load_config
from landcover.training import run_experiment


if __name__ == "__main__":
    cfg = load_config(str(ROOT / "configs" / "loveda.yaml"))
    run_experiment(cfg, tag="loveda")
