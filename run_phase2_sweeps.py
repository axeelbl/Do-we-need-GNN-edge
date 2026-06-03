from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, GridConfig, ModelConfig, TrainingConfig
from main import run

RESULTS_DIR = SRC_DIR / "results" / "phase2_sweeps"
SUMMARY_CSV = RESULTS_DIR / "phase2_all_runs.csv"
SUMMARY_XLSX = PROJECT_ROOT / "informe_fase2_sweeps.xlsx"
PLOTS_DIR = PROJECT_ROOT / "graficas_tfg" / "fase2_sweeps"

DATA_FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]
SEEDS = [42, 43, 44]
EPOCHS = 50
PHYSICS_LAMBDA = 0.1

MODEL_LABELS = {
    "mlp_baseline": "MLP",
    "full_gnn": "FullGNN",
    "tiny_gnn": "TinyGNN",
    "tiny_gnn_pinn": "TinyGNN + PINN",
}
MODEL_COLORS = {
    "mlp_baseline": "#4C78A8",
    "full_gnn": "#F58518",
    "tiny_gnn": "#54A24B",
    "tiny_gnn_pinn": "#E45756",
}
MODEL_ORDER = ["mlp_baseline", "full_gnn", "tiny_gnn", "tiny_gnn_pinn"]


def build_config(seed: int) -> object:
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    start_all = time.perf_counter()

    experiments = []
    for data_fraction in DATA_FRACTIONS:
        experiments.append({"sweep": "data_fraction", "data_fraction": data_fraction, "noise_level": 0.0})
    for noise_level in NOISE_LEVELS:
        experiments.append({"sweep": "noise_level", "data_fraction": 1.0, "noise_level": noise_level})

    total_runs = len(experiments) * len(SEEDS)
    run_idx = 0

    for exp in experiments:
        for seed in SEEDS:
            run_idx += 1
            cfg = build_config(seed)
            run_label = (
                f"{exp['sweep']}_df{exp['data_fraction']}_noise{exp['noise_level']}_seed{seed}"
                .replace(".", "p")
            )
            print(f"\n\n===== PHASE 2 RUN {run_idx}/{total_runs}: {run_label} =====", flush=True)
            t0 = time.perf_counter()
            results = run(
                config=cfg,
                results_dir=RESULTS_DIR / run_label,
                plots_dir=RESULTS_DIR / run_label / "plots",
                run_label=run_label,
                data_fraction=exp["data_fraction"],
                noise_level=exp["noise_level"],
            )
            elapsed = time.perf_counter() - t0

            for model_name, metrics in results.items():
                rows.append({
                    "sweep": exp["sweep"],
                    "run_label": run_label,
                    "seed": seed,
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "data_fraction": exp["data_fraction"],
                    "noise_level": exp["noise_level"],
                    "epochs": EPOCHS,
                    "physics_lambda": PHYSICS_LAMBDA if model_name == "tiny_gnn_pinn" else 0.0,
                    "runtime_seconds": round(elapsed, 3),
                    **metrics,
                })

            pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    all_df = pd.DataFrame(rows)
    agg_df = aggregate_results(all_df)
    save_report(all_df, agg_df)
    save_phase2_plots(agg_df)

    print("\n===== PHASE 2 FINISHED =====")
    print(f"CSV:   {SUMMARY_CSV}")
    print(f"Excel: {SUMMARY_XLSX}")
    print(f"Plots: {PLOTS_DIR}")
    print(f"Total time: {time.perf_counter() - start_all:.1f}s")


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sweep", "model", "model_label", "data_fraction", "noise_level"]
    metric_cols = [
        "val_mse",
        "test_mse",
        "test_physics_violation",
        "num_parameters",
        "model_size_bytes",
        "inference_time",
    ]
    agg = df.groupby(group_cols, as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metric_cols},
        **{f"{metric}_std": (metric, "std") for metric in metric_cols},
        n=("seed", "nunique"),
    )
    for metric in metric_cols:
        agg[f"{metric}_sem"] = agg[f"{metric}_std"].fillna(0.0) / np.sqrt(agg["n"].clip(lower=1))
        agg[f"{metric}_ci95"] = 1.96 * agg[f"{metric}_sem"]
    return agg


def save_report(all_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="Runs", index=False)
        agg_df.to_excel(writer, sheet_name="Aggregated", index=False)
        notes = pd.DataFrame([
            {"apartado": "Objetivo", "texto": "Fase 2: estudiar degradación por cantidad de datos y ruido con varias seeds."},
            {"apartado": "Data fraction", "texto": "Se entrena con [1.0, 0.5, 0.2, 0.1, 0.05, 0.01], manteniendo noise_level=0."},
            {"apartado": "Noise level", "texto": "Se añade ruido gaussiano solo al entrenamiento con [0.0, 0.01, 0.05, 0.1, 0.2], manteniendo data_fraction=1."},
            {"apartado": "Seeds", "texto": f"Cada punto usa seeds {SEEDS}; las gráficas muestran media ± IC95."},
            {"apartado": "Aviso", "texto": "Fase 2 usa una configuración ligera para iterar rápido; la fase final debe subir epochs/seeds si el tiempo lo permite."},
        ])
        notes.to_excel(writer, sheet_name="Notas", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 70)


def save_phase2_plots(agg_df: pd.DataFrame) -> None:
    _line_plot(
        agg_df[agg_df["sweep"] == "data_fraction"].copy(),
        x="data_fraction",
        y="test_mse",
        title="Fase 2: Test MSE vs cantidad de datos",
        xlabel="data_fraction",
        filename="01_mse_vs_data_fraction.png",
        invert_x=True,
    )
    _line_plot(
        agg_df[agg_df["sweep"] == "noise_level"].copy(),
        x="noise_level",
        y="test_mse",
        title="Fase 2: Test MSE vs ruido de entrenamiento",
        xlabel="noise_level",
        filename="02_mse_vs_noise_level.png",
    )
    _line_plot(
        agg_df[agg_df["sweep"] == "data_fraction"].copy(),
        x="data_fraction",
        y="test_physics_violation",
        title="Fase 2: Violación física vs cantidad de datos",
        xlabel="data_fraction",
        filename="03_physics_vs_data_fraction.png",
        invert_x=True,
    )
    _line_plot(
        agg_df[agg_df["sweep"] == "noise_level"].copy(),
        x="noise_level",
        y="test_physics_violation",
        title="Fase 2: Violación física vs ruido de entrenamiento",
        xlabel="noise_level",
        filename="04_physics_vs_noise_level.png",
    )
    _resource_plot(agg_df)


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, filename: str, invert_x: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values(x)
        if sub.empty:
            continue
        ax.errorbar(
            sub[x],
            sub[f"{y}_mean"],
            yerr=sub[f"{y}_ci95"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{y} mean ± IC95")
    ax.set_yscale("log")
    if invert_x:
        ax.set_xscale("log")
        ax.invert_xaxis()
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resource_plot(agg_df: pd.DataFrame) -> None:
    sub = agg_df[(agg_df["sweep"] == "data_fraction") & (agg_df["data_fraction"] == 1.0)].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model in MODEL_ORDER:
        row = sub[sub["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        ax.scatter(
            row["num_parameters_mean"],
            row["test_mse_mean"],
            s=120,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.9,
            edgecolor="white",
        )
        ax.annotate(MODEL_LABELS[model], (row["num_parameters_mean"], row["test_mse_mean"]), xytext=(7, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fase 2: trade-off precisión / parámetros")
    ax.set_xlabel("Nº parámetros")
    ax.set_ylabel("Test MSE medio")
    ax.grid(alpha=0.3, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_tradeoff_precision_parametros.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
