
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, SWEEP_HIDDEN_DIMS, SWEEP_RESULTS_DIR
from main import run
from utils.metrics import save_metrics, save_metrics_dataframe


def main() -> None:

    args = _parse_args()
    sweep_dir = Path(args.output_dir) if args.output_dir else SWEEP_RESULTS_DIR
    hidden_dims = args.hidden_dims

    print(f"Executant {len(hidden_dims)} configuracions de hidden_dim:")
    print(f"  {hidden_dims}")

    rows: list[dict[str, Any]] = []

    for idx, hidden_dim in enumerate(hidden_dims, start=1):
        run_label = f"sweep_{idx:02d}_h{hidden_dim}"
        run_dir = sweep_dir / run_label

        print(f"\n[{idx}/{len(hidden_dims)}] {run_label}")

        # @TODO (TASCA 4): Modifiqueu run() per acceptar hidden_dim
        # i mesureu test_mse, test_physics_violation, num_parameters
        # per a cada mida.

        results = run(
            config=CONFIG,
            results_dir=run_dir,
            plots_dir=run_dir / "plots",
            run_label=run_label,
        )

        for model_name, metrics in results.items():
            rows.append({
                "run_label": run_label,
                "hidden_dim": hidden_dim,
                "model": model_name,
                **metrics,
            })

    summary_df = pd.DataFrame(rows)
    summary_csv = sweep_dir / "sweep_metrics.csv"

    save_metrics_dataframe(summary_csv, summary_df)

    print(f"\nSweep completat.")
    print(f"Resultats: {summary_csv}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barrido de mides d'embedding (tasca 4 del TFG)."
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=SWEEP_HIDDEN_DIMS,
        help="Mides d'embedding a provar.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta on guardar resultats.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
