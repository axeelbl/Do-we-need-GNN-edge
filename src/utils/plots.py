from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import torch
from matplotlib import pyplot as plt

MODEL_COLORS = {
    "mlp_baseline":  "#6c7a89",
    "full_gnn":      "#2ecc71",
    "tiny_gnn":      "#3498db",
    "tiny_gnn_pinn": "#e74c3c", 
}
MODEL_LABELS = {
    "mlp_baseline":  "MLP",
    "full_gnn":      "FullGNN",
    "tiny_gnn":      "TinyGNN",
    "tiny_gnn_pinn": "TinyGNN + PINN",
}

def save_all_plots(
    dataframe:      pd.DataFrame,
    loss_histories: dict[str, list[dict]],
    grid_size:      int,
    snapshots:      dict[str, dict[str, torch.Tensor]] | None = None,
    output_dir:     Path = Path("results/plots"),
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_bar_comparison(dataframe, "test_mse",               "Test MSE (error de predicció)",        output_dir)
    _save_bar_comparison(dataframe, "test_physics_violation",  "Violació física (residual MSE)",       output_dir)
    _save_bar_comparison(dataframe, "num_parameters",          "Nombre de paràmetres entrenables",     output_dir)
    _save_bar_comparison(dataframe, "inference_time",          "Temps d'inferència (s)",               output_dir)
    _save_tradeoff_chart(dataframe, output_dir)
    _save_loss_curves(loss_histories, output_dir)

    if snapshots:
        _save_grid_snapshots(snapshots, grid_size, output_dir)


def _save_bar_comparison(
    df:         pd.DataFrame,
    metric:     str,
    title:      str,
    output_dir: Path,
) -> None:
    if metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    models  = df["model"].tolist()
    values  = df[metric].tolist()
    colors  = [MODEL_COLORS.get(m, "#aaa") for m in models]
    labels  = [MODEL_LABELS.get(m, m)      for m in models]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.4f}" if val < 10 else f"{val:.0f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png", dpi=150)
    plt.close(fig)


def _save_tradeoff_chart(df: pd.DataFrame, output_dir: Path) -> None:
    required = {"model", "test_mse", "test_physics_violation", "num_parameters"}
    if not required.issubset(df.columns):
        return

    fig, ax    = plt.subplots(figsize=(7, 5))
    max_params = max(float(df["num_parameters"].max()), 1.0)

    for _, row in df.iterrows():
        m     = str(row["model"])
        color = MODEL_COLORS.get(m, "#aaa")
        label = MODEL_LABELS.get(m, m)
        size  = 80 + 420 * (float(row["num_parameters"]) / max_params)

        ax.scatter(row["test_physics_violation"], row["test_mse"],
                   s=size, color=color, alpha=0.85,
                   edgecolors="white", linewidth=0.8, label=label)
        ax.annotate(label, (row["test_physics_violation"], row["test_mse"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_ylabel("Error de predicció (test MSE)",  fontsize=10)
    ax.set_title("Error de predicció vs Violació física", fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "tradeoff_mse_vs_physics.png", dpi=150)
    plt.close(fig)


def _save_loss_curves(
    loss_histories: dict[str, list[dict]],
    output_dir:     Path,
) -> None:
    
    fig, ax = plt.subplots(figsize=(8, 4))

    for model_name, history in loss_histories.items():
        color  = MODEL_COLORS.get(model_name, "#aaa")
        label  = MODEL_LABELS.get(model_name, model_name)
        epochs = list(range(1, len(history) + 1))
        losses = [h["data_loss"] for h in history]
        ax.plot(epochs, losses, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("epochs")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def _save_grid_snapshots(
    snapshots:  dict[str, dict[str, torch.Tensor]],
    grid_size:  int,
    output_dir: Path,
) -> None:
    model_names = list(snapshots.keys())
    num_cols    = len(model_names) + 1

    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    gt        = snapshots[model_names[0]]["ground_truth"].reshape(grid_size, grid_size).numpy()
    vmin, vmax = float(gt.min()), float(gt.max())

    im = axes[0].imshow(gt, cmap="hot", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth", fontsize=11)
    axes[0].axis("off")

    for idx, name in enumerate(model_names):
        pred  = snapshots[name]["prediction"].reshape(grid_size, grid_size).detach().numpy()
        label = MODEL_LABELS.get(name, name)
        mse   = float(((pred - gt) ** 2).mean())
        axes[idx + 1].imshow(pred, cmap="hot", vmin=vmin, vmax=vmax)
        axes[idx + 1].set_title(f"{label}\nMSE={mse:.4f}", fontsize=10)
        axes[idx + 1].axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Temperatura")
    fig.suptitle("Grid de temperatura — últim pas de test", fontsize=12, y=1.02)
    fig.savefig(output_dir / "grid_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_embedding_sweep_plot(
    sweep_df:   pd.DataFrame,
    output_dir: Path,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_name in ["full_gnn", "tiny_gnn"]:
        sub    = sweep_df[sweep_df["model"] == model_name]
        if sub.empty:
            continue
        color  = MODEL_COLORS.get(model_name, "#aaa")
        label  = MODEL_LABELS.get(model_name, model_name)
        marker = "o" if model_name == "tiny_gnn" else "s"

        axes[0].plot(sub["hidden_dim"], sub["test_mse"],
                     f"{marker}-", color=color, label=label, linewidth=1.8)
        axes[1].plot(sub["hidden_dim"], sub["test_physics_violation"],
                     f"{marker}-", color=color, label=label, linewidth=1.8)

    for ax, ylabel, title in zip(
        axes,
        ["Test MSE", "Violació física (MSE)"],
        ["Error de predicció vs mida embedding",
         "Violació física vs mida embedding"],
    ):
        ax.set_xlabel("Embedding size (hidden_dim)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "embedding_size_sweep.png", dpi=150)
    plt.close(fig)
