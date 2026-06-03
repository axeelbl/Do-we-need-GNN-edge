from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONFIG, AppConfig, ModelConfig, TrainingConfig
from main import run

REPORT_PATH = PROJECT_ROOT / "informe_ejecuciones_tfg.xlsx"
BASE_RESULTS = SRC_DIR / "results" / "experimentos_excel"

EXPERIMENTS = [
    {"id": "exp_01_base_100ep", "descripcion": "Configuración base reducida a 100 epochs para comparación rápida.", "epochs": 100, "hidden_dim_full": 64, "hidden_dim_tiny": 16, "data_fraction": 1.0, "noise_level": 0.0},
    {"id": "exp_02_modelos_medios_100ep", "descripcion": "Reduce capacidad: FullGNN hidden=32 y TinyGNN hidden=8.", "epochs": 100, "hidden_dim_full": 32, "hidden_dim_tiny": 8, "data_fraction": 1.0, "noise_level": 0.0},
    {"id": "exp_03_modelos_pequenos_100ep", "descripcion": "Reduce bastante la capacidad: FullGNN hidden=16 y TinyGNN hidden=4.", "epochs": 100, "hidden_dim_full": 16, "hidden_dim_tiny": 4, "data_fraction": 1.0, "noise_level": 0.0},
]


def build_config(exp: dict) -> AppConfig:
    return replace(
        CONFIG,
        model=ModelConfig(hidden_dim_full=exp["hidden_dim_full"], hidden_dim_tiny=exp["hidden_dim_tiny"], input_dim=1, output_dim=1),
        training=TrainingConfig(
            epochs=exp["epochs"],
            learning_rate=CONFIG.training.learning_rate,
            physics_lambda=CONFIG.training.physics_lambda,
            seed=CONFIG.training.seed,
            device=CONFIG.training.device,
        ),
        results_dir=BASE_RESULTS / exp["id"],
        plots_dir=BASE_RESULTS / exp["id"] / "plots",
    )


def main() -> None:
    result_rows = []
    config_rows = []
    start_all = time.perf_counter()

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        print(f"\n\n===== EJECUCIÓN {idx}/{len(EXPERIMENTS)}: {exp['id']} =====")
        cfg = build_config(exp)
        t0 = time.perf_counter()
        results = run(config=cfg, results_dir=cfg.results_dir, plots_dir=cfg.plots_dir, run_label=exp["id"], data_fraction=exp["data_fraction"], noise_level=exp["noise_level"])
        elapsed = time.perf_counter() - t0

        config_rows.append({
            "experiment_id": exp["id"], "descripcion": exp["descripcion"], "grid_size": cfg.grid.grid_size,
            "num_timesteps": cfg.grid.num_timesteps, "train_steps": cfg.grid.train_steps, "val_steps": cfg.grid.val_steps, "test_steps": cfg.grid.test_steps,
            "alpha": cfg.grid.alpha, "dt": cfg.grid.dt, "dx": cfg.grid.dx, "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate, "physics_lambda": cfg.training.physics_lambda, "device": cfg.training.device,
            "hidden_dim_full": cfg.model.hidden_dim_full, "hidden_dim_tiny": cfg.model.hidden_dim_tiny,
            "data_fraction": exp["data_fraction"], "noise_level": exp["noise_level"],
            "runtime_seconds_total_experiment": round(elapsed, 3),
        })

        for model_name, metrics in results.items():
            result_rows.append({
                "experiment_id": exp["id"], "model": model_name, "descripcion": exp["descripcion"],
                "epochs": cfg.training.epochs, "hidden_dim_full": cfg.model.hidden_dim_full, "hidden_dim_tiny": cfg.model.hidden_dim_tiny,
                "grid_size": cfg.grid.grid_size, "num_parameters": metrics.get("num_parameters"),
                "model_size_bytes": metrics.get("model_size_bytes"), "val_mse": metrics.get("val_mse"),
                "test_mse": metrics.get("test_mse"), "test_physics_violation": metrics.get("test_physics_violation"),
                "inference_time_seconds": metrics.get("inference_time"), "data_fraction": metrics.get("data_fraction"),
                "noise_level": metrics.get("noise_level"), "physics_lambda": metrics.get("physics_lambda"),
                "train_pairs": metrics.get("train_pairs"), "val_pairs": metrics.get("val_pairs"), "test_pairs": metrics.get("test_pairs"),
                "runtime_seconds_total_experiment": round(elapsed, 3),
            })

    results_df = pd.DataFrame(result_rows)
    config_df = pd.DataFrame(config_rows)
    explanation_df = pd.DataFrame([
        {"apartado": "Objetivo", "texto": "Comparar MLP baseline, FullGNN, TinyGNN y TinyGNN+PINN en difusión de calor con tres tamaños de modelo."},
        {"apartado": "Cómo leerlo", "texto": "Menor test_mse indica mejor predicción. Menor test_physics_violation indica mayor coherencia física. Menos parámetros indica modelo más ligero."},
        {"apartado": "Conclusión inicial", "texto": "En una simulación limpia, todos aprenden bien; lo interesante es comparar precisión frente a parámetros y tiempo."},
        {"apartado": "Aviso", "texto": "data_fraction reduce realmente los pares temporales de entrenamiento, noise_level añade ruido gaussiano solo al entrenamiento y TinyGNN+PINN usa Loss_data + lambda*Loss_physics."},
    ])

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Resultados", index=False)
        config_df.to_excel(writer, sheet_name="Configuracion", index=False)
        explanation_df.to_excel(writer, sheet_name="Explicacion", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 60)

    elapsed_all = time.perf_counter() - start_all
    print("\n\n===== RESUMEN FINAL =====")
    print(results_df.to_string(index=False))
    print(f"\nExcel generado: {REPORT_PATH}")
    print(f"Tiempo total: {elapsed_all:.1f}s")

if __name__ == "__main__":
    main()
