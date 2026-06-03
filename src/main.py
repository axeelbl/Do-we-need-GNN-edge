from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import AppConfig, CONFIG
from data.graph_builder import build_grid_graph
from data.heat_diffusion import add_gaussian_noise, build_temporal_pairs, select_data_fraction, simulate_heat_diffusion
from models.baselines import MLPBaseline
from models.full_gnn import FullGNN
from models.tiny_gnn import TinyGNN
from training.evaluator import (
    evaluate_physics_violation,
    evaluate_prediction_error,
    measure_inference_time,
    measure_model_size_bytes,
)
from training.trainer import train_one_epoch
from utils.metrics import (
    build_initial_metrics,
    build_metrics_dataframe,
    count_trainable_parameters,
    save_metrics,
    save_metrics_dataframe,
)
from utils.plots import save_all_plots
from utils.seed import set_seed


def run(
    config: AppConfig = CONFIG,
    results_dir: Path | None = None,
    plots_dir: Path | None = None,
    run_label: str = "baseline",
    data_fraction: float = 1.0,
    noise_level: float = 0.0,
) -> dict[str, dict[str, Any]]:

    set_seed(config.training.seed)

    results_dir = results_dir or config.results_dir
    plots_dir = plots_dir or config.plots_dir
    device = config.training.device

    print(f"\n{'=' * 70}")
    print(f"  Experiment: {run_label}")
    print(f"  Grid: {config.grid.grid_size}×{config.grid.grid_size} | "
          f"data_fraction: {data_fraction} | noise: {noise_level:.4f}")
    print(f"{'=' * 70}")

    print("\n[1/4] Simulant difusió de calor...")
    sim = simulate_heat_diffusion(
        grid_size=config.grid.grid_size,
        num_timesteps=config.grid.num_timesteps,
        train_steps=config.grid.train_steps,
        val_steps=config.grid.val_steps,
        test_steps=config.grid.test_steps,
        alpha=config.grid.alpha,
        dt=config.grid.dt,
        dx=config.grid.dx,
        seed=config.training.seed,
    )

    x_train_in, x_train_tgt = build_temporal_pairs(sim.train_states)
    x_val_in, x_val_tgt = build_temporal_pairs(sim.val_states)
    x_test_in, x_test_tgt = build_temporal_pairs(sim.test_states)

    x_train_in, x_train_tgt = select_data_fraction(
        x_train_in,
        x_train_tgt,
        fraction=data_fraction,
        seed=config.training.seed,
    )

    if noise_level > 0.0:
        x_train_in = add_gaussian_noise(x_train_in, sigma=noise_level, seed=config.training.seed + 17)
        x_train_tgt = add_gaussian_noise(x_train_tgt, sigma=noise_level, seed=config.training.seed + 31)

    x_train_in  = x_train_in.unsqueeze(-1).to(device)
    x_train_tgt = x_train_tgt.unsqueeze(-1).to(device)
    x_val_in    = x_val_in.unsqueeze(-1).to(device)
    x_val_tgt   = x_val_tgt.unsqueeze(-1).to(device)
    x_test_in   = x_test_in.unsqueeze(-1).to(device)
    x_test_tgt  = x_test_tgt.unsqueeze(-1).to(device)

    print("[2/4] Construint graf de veïnatge...")
    graph = build_grid_graph(grid_size=config.grid.grid_size)
    edge_index = graph.edge_index.to(device)

    print("[3/4] Entrenant models...")

    models: dict[str, nn.Module] = {
        "mlp_baseline": MLPBaseline(hidden_dim=config.model.hidden_dim_full),
        "full_gnn":     FullGNN(hidden_dim=config.model.hidden_dim_full),
        "tiny_gnn":     TinyGNN(hidden_dim=config.model.hidden_dim_tiny),
        "tiny_gnn_pinn": TinyGNN(hidden_dim=config.model.hidden_dim_tiny),
    }

    loss_fn = nn.MSELoss()
    loss_histories: dict[str, list[dict]] = {name: [] for name in models}

    for model_name, model in models.items():
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
        )
        print(f"  Entrenant {model_name:15s} ({count_trainable_parameters(model):6d} params)...")

        for epoch in range(1, config.training.epochs + 1):
            physics_lambda = config.training.physics_lambda if model_name == "tiny_gnn_pinn" else 0.0
            losses = train_one_epoch(
                model=model,
                x_input=x_train_in,
                x_target=x_train_tgt,
                edge_index=edge_index,
                optimizer=optimizer,
                loss_fn=loss_fn,
                physics_lambda=physics_lambda,
                grid_size=config.grid.grid_size,
                alpha=config.grid.alpha,
                dt=config.grid.dt,
                dx=config.grid.dx,
            )
            loss_histories[model_name].append(losses)

            if epoch % max(1, config.training.epochs // 5) == 0 or epoch == 1:
                print(
                    f"    Epoca {epoch:>4d}/{config.training.epochs} | "
                    f"loss={losses['total_loss']:.5f} "
                    f"(data={losses['data_loss']:.5f}, phys={losses['physics_loss']:.5f})"
                )

    print("[4/4] Avaluant models...")

    results: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, torch.Tensor]] = {}

    for model_name, model in models.items():
        model.eval()

        val_mse = evaluate_prediction_error(model, x_val_in, x_val_tgt, edge_index)
        test_mse = evaluate_prediction_error(model, x_test_in, x_test_tgt, edge_index)
        phys_viol = evaluate_physics_violation(
            model,
            x_test_in,
            edge_index,
            alpha=config.grid.alpha,
            dt=config.grid.dt,
            dx=config.grid.dx,
            grid_size=config.grid.grid_size,
        )
        infer_time = measure_inference_time(model, x_test_in, edge_index)
        num_params = count_trainable_parameters(model)
        model_size = measure_model_size_bytes(model)

        train_loss = loss_histories[model_name][-1]["total_loss"]

        results[model_name] = {
            "num_parameters": num_params,
            "model_size_bytes": model_size,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "test_physics_violation": phys_viol,
            "inference_time": infer_time,
            "data_fraction": data_fraction,
            "noise_level": noise_level,
            "physics_lambda": config.training.physics_lambda if model_name == "tiny_gnn_pinn" else 0.0,
            "train_pairs": int(x_train_in.shape[0]),
            "val_pairs": int(x_val_in.shape[0]),
            "test_pairs": int(x_test_in.shape[0]),
        }

        with torch.no_grad():
            from training.trainer import model_forward
            x_snap = x_test_in[-1]
            pred_snap = model_forward(model, x_snap, edge_index)

        snapshots[model_name] = {
            "ground_truth": x_test_tgt[-1].squeeze(-1).cpu(),
            "prediction": pred_snap.squeeze(-1).cpu(),
        }

        print(
            f"  {model_name:15s} | params={num_params:6d} | "
            f"val_mse={val_mse:.5f} | test_mse={test_mse:.5f} | phys_viol={phys_viol:.5f} | "
            f"time={infer_time:.4f}s"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = results_dir / "metrics.json"
    metrics_csv = results_dir / "metrics.csv"

    initial_metrics = build_initial_metrics(config)
    initial_metrics["runs"] = [{"label": run_label, "models": results}]
    save_metrics(metrics_json, initial_metrics)

    df = build_metrics_dataframe(results)
    save_metrics_dataframe(metrics_csv, df)

    save_all_plots(
        dataframe=df,
        loss_histories=loss_histories,
        grid_size=config.grid.grid_size,
        snapshots=snapshots,
        output_dir=plots_dir,
    )

    print(f"\n  Resultats guardats a:  {results_dir}")
    print(f"  Gràfics guardats a:    {plots_dir}")

    return results


if __name__ == "__main__":
    run()
