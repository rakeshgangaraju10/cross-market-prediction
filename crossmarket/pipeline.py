"""End-to-end run: data -> spillovers -> models -> results/ (used by the web app)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_closes, log_returns, realized_volatility
from .forecasting import HORIZONS, run_horizon
from .spillover import generalized_fevd, spillover_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(horizons=HORIZONS, neural: bool = True, epochs: int = 300,
        out: Path = RESULTS_DIR) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    returns = log_returns(load_closes())
    rv = realized_volatility(returns)

    fevd = generalized_fevd(np.log(rv))
    table = spillover_table(fevd)
    fevd.to_csv(out / "spillover_fevd.csv")
    print(f"Total spillover index (full sample): {table['total']:.1f}%")

    all_metrics, histories, splits = [], {}, {}
    for h in horizons:
        print(f"Horizon {h}d ...")
        res = run_horizon(returns, h, neural=neural, epochs=epochs)
        all_metrics.append(res["metrics"])
        histories[str(h)] = res["history"]
        splits[str(h)] = res["split_dates"]
        preds = pd.concat(res["predictions"], axis=1)
        preds.to_csv(out / f"predictions_h{h}.csv")
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "history.json").write_text(json.dumps(histories))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--no-neural", action="store_true", help="skip MLP/GNN (no torch needed)")
    args = ap.parse_args()
    m = run(args.horizons, not args.no_neural, args.epochs)
    summary = m[m.market == "All"].pivot(index="model", columns="horizon", values="QLIKE")
    print("\nTest-set QLIKE (lower is better):")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
