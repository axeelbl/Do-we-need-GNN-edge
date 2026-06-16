from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, GridConfig, ModelConfig, TrainingConfig
from main import run

RESULTS_DIR = SRC_DIR / "results" / "data_fraction_gpu"
SUMMARY_CSV = RESULTS_DIR / "data_fraction_gpu_runs.csv"
SUMMARY_XLSX = PROJECT_ROOT / "informe_data_fraction_gpu.xlsx"

DATA_FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
SEEDS = [42, 43, 44]
EPOCHS = 50
PHYSICS_LAMBDA = 0.1

MODEL_LABELS = {
    "mlp_baseline": "MLP",
    "full_gnn": "FullGNN",
    "tiny_gnn": "TinyGNN",
    "tiny_gnn_pinn": "TinyGNN + PINN",
}


def build_config(seed: int):
    return replace(
        CONFIG,
        grid=GridConfig(
            grid_size=16,
            num_timesteps=80,
            train_steps=48,
            val_steps=16,
            test_steps=16,
            alpha=CONFIG.grid.alpha,
            dt=CONFIG.grid.dt,
            dx=CONFIG.grid.dx,
        ),
        model=ModelConfig(hidden_dim_full=32, hidden_dim_tiny=8),
        training=TrainingConfig(
            epochs=EPOCHS,
            learning_rate=CONFIG.training.learning_rate,
            physics_lambda=PHYSICS_LAMBDA,
            seed=seed,
            device=CONFIG.training.device,
        ),
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["val_mse", "test_mse", "test_physics_violation", "inference_time", "num_parameters", "model_size_bytes"]
    agg = df.groupby(["data_fraction", "model", "model_label"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_cols},
        **{f"{m}_std": (m, "std") for m in metric_cols},
        n=("seed", "nunique"),
    )
    return agg.sort_values(["data_fraction", "model"], ascending=[False, True])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(DATA_FRACTIONS) * len(SEEDS)
    idx = 0
    start_all = time.perf_counter()
    print(f"DEVICE={CONFIG.training.device}", flush=True)

    for data_fraction in DATA_FRACTIONS:
        for seed in SEEDS:
            idx += 1
            cfg = build_config(seed)
            run_label = f"df{data_fraction}_seed{seed}".replace(".", "p")
            print(f"\n===== DATA FRACTION RUN {idx}/{total}: {run_label} =====", flush=True)
            t0 = time.perf_counter()
            results = run(
                config=cfg,
                results_dir=RESULTS_DIR / run_label,
                plots_dir=RESULTS_DIR / run_label / "plots",
                run_label=run_label,
                data_fraction=data_fraction,
                noise_level=0.0,
            )
            elapsed = time.perf_counter() - t0
            for model, metrics in results.items():
                rows.append({
                    "run_label": run_label,
                    "seed": seed,
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "data_fraction": data_fraction,
                    "epochs": EPOCHS,
                    "physics_lambda": PHYSICS_LAMBDA if model == "tiny_gnn_pinn" else 0.0,
                    "runtime_seconds": round(elapsed, 3),
                    **metrics,
                })
            pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    all_df = pd.DataFrame(rows)
    agg_df = aggregate(all_df)
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="Runs", index=False)
        agg_df.to_excel(writer, sheet_name="Aggregated", index=False)
        notes = pd.DataFrame([
            {"apartado": "Objetivo", "texto": "Sweep de data_fraction con seeds 42,43,44 en GPU."},
            {"apartado": "Config", "texto": f"data_fraction={DATA_FRACTIONS}, epochs={EPOCHS}, physics_lambda={PHYSICS_LAMBDA}, device={CONFIG.training.device}"},
            {"apartado": "Lectura", "texto": "Menor test_mse = mejor prediccion; menor test_physics_violation = mejor coherencia fisica."},
        ])
        notes.to_excel(writer, sheet_name="Notas", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 70)

    print("\n===== DATA FRACTION GPU FINISHED =====", flush=True)
    print(f"CSV: {SUMMARY_CSV}", flush=True)
    print(f"Excel: {SUMMARY_XLSX}", flush=True)
    print(f"Total time: {time.perf_counter() - start_all:.1f}s", flush=True)
    print("\nAGGREGATED:", flush=True)
    print(agg_df[["data_fraction", "model_label", "test_mse_mean", "test_mse_std", "test_physics_violation_mean", "n"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
