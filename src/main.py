"""Pipeline experimental: baselines, GNNs, metricas y graficos."""

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from config import (
    DEVICE,
    EPOCHS,
    CONTROLLED_TEST_SUBSET_SIZE,
    CONTROLLED_TRAIN_SUBSET_SIZE,
    FULL_GNN_TEST_SUBSET_SIZE,
    FULL_GNN_TRAIN_SUBSET_SIZE,
    IMAGES_PER_NODE,
    K_NEIGHBORS,
    LEARNING_RATE,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    RUN_MODE,
    TINY_GNN_TEST_SUBSET_SIZE,
    TINY_GNN_TRAIN_SUBSET_SIZE,
)
from data.cifar_loader import CifarData, load_cifar10
from data.feature_extractor import build_node_features_from_images
from data.graph_builder import build_similarity_graph
from models.baselines import RandomBaseline, SimpleMLPBaseline
from models.full_gnn import FullGNN
from models.tiny_gnn import TinyGNN
from training.evaluator import evaluate_accuracy, measure_inference_time
from training.trainer import train_one_epoch
from utils.metrics import (
    build_metrics_dataframe,
    count_trainable_parameters,
    save_metrics,
    save_metrics_dataframe,
)
from utils.plots import save_metrics_bar_charts
from utils.seed import set_seed


def run(
    run_mode: str = RUN_MODE,
    k_neighbors: int = K_NEIGHBORS,
    epochs: int = EPOCHS,
    images_per_node: int = IMAGES_PER_NODE,
    results_dir: Any = RESULTS_DIR,
    plots_dir: Any = PLOTS_DIR,
    run_label: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Ejecuta todos los modelos definidos para el modo actual."""

    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)

    experiments = _build_experiments(run_mode)
    graph_cache: dict[tuple[int | None, int | None, str], tuple[Data, CifarData]] = {}
    results: dict[str, dict[str, Any]] = {}

    for model_name, experiment in experiments.items():
        cache_key = (
            experiment["train_subset_size"],
            experiment["test_subset_size"],
            experiment["dataset_mode"],
        )

        if cache_key not in graph_cache:
            graph_cache[cache_key] = _load_graph_for_experiment(
                train_subset_size=experiment["train_subset_size"],
                test_subset_size=experiment["test_subset_size"],
                device=device,
                k_neighbors=k_neighbors,
                images_per_node=images_per_node,
            )

        graph, cifar_data = graph_cache[cache_key]
        _print_graph_summary(
            graph=graph,
            cifar_data=cifar_data,
            model_name=model_name,
            dataset_mode=experiment["dataset_mode"],
            run_mode=run_mode,
            k_neighbors=k_neighbors,
            images_per_node=images_per_node,
        )

        results[model_name] = _run_model(
            model=experiment["model"],
            graph=graph,
            device=device,
            model_type=experiment["model_type"],
            dataset_mode=experiment["dataset_mode"],
            train_subset_size=experiment["train_subset_size"],
            test_subset_size=experiment["test_subset_size"],
            run_mode=run_mode,
            k_neighbors=k_neighbors,
            epochs=epochs,
            images_per_node=images_per_node,
            run_label=run_label,
        )

    metrics_dataframe = build_metrics_dataframe(results)

    metrics_path = results_dir / "metrics.json"
    metrics_csv_path = results_dir / "metrics.csv"
    save_metrics(metrics_path, results)
    save_metrics_dataframe(metrics_csv_path, metrics_dataframe)
    save_metrics_bar_charts(metrics_dataframe, plots_dir)

    _print_results_summary(results)
    print(f"Metricas guardadas en: {metrics_path}")
    print(f"DataFrame guardado en: {metrics_csv_path}")
    print(f"Graficos guardados en: {plots_dir}")

    return results


def _build_experiments(run_mode: str) -> dict[str, dict[str, Any]]:
    """Define que datos usa cada modelo segun RUN_MODE."""

    if run_mode not in {"resource_efficiency", "controlled_subset"}:
        raise ValueError("RUN_MODE debe ser 'resource_efficiency' o 'controlled_subset'.")

    if run_mode == "controlled_subset":
        common_train_subset: int | None = CONTROLLED_TRAIN_SUBSET_SIZE
        common_test_subset: int | None = CONTROLLED_TEST_SUBSET_SIZE
        return {
            "random_baseline": _experiment(
                RandomBaseline(), "random", "controlled_subset", common_train_subset, common_test_subset
            ),
            "mlp_baseline": _experiment(
                SimpleMLPBaseline(), "no_gnn", "controlled_subset", common_train_subset, common_test_subset
            ),
            "tiny_gnn": _experiment(
                TinyGNN(), "tiny_gnn", "controlled_subset", common_train_subset, common_test_subset
            ),
            "full_gnn": _experiment(
                FullGNN(), "full_gnn", "controlled_subset", common_train_subset, common_test_subset
            ),
        }

    return {
        "random_baseline": _experiment(
            RandomBaseline(),
            "random",
            "resource_subset",
            TINY_GNN_TRAIN_SUBSET_SIZE,
            TINY_GNN_TEST_SUBSET_SIZE,
        ),
        "mlp_baseline": _experiment(
            SimpleMLPBaseline(),
            "no_gnn",
            "resource_subset",
            TINY_GNN_TRAIN_SUBSET_SIZE,
            TINY_GNN_TEST_SUBSET_SIZE,
        ),
        "tiny_gnn": _experiment(
            TinyGNN(),
            "tiny_gnn",
            "resource_subset",
            TINY_GNN_TRAIN_SUBSET_SIZE,
            TINY_GNN_TEST_SUBSET_SIZE,
        ),
        "full_gnn": _experiment(
            FullGNN(),
            "full_gnn",
            "full_cifar10",
            FULL_GNN_TRAIN_SUBSET_SIZE,
            FULL_GNN_TEST_SUBSET_SIZE,
        ),
    }


def _experiment(
    model: nn.Module,
    model_type: str,
    dataset_mode: str,
    train_subset_size: int | None,
    test_subset_size: int | None,
) -> dict[str, Any]:
    """Crea una definicion pequena de experimento."""

    return {
        "model": model,
        "model_type": model_type,
        "dataset_mode": dataset_mode,
        "train_subset_size": train_subset_size,
        "test_subset_size": test_subset_size,
    }


def _load_graph_for_experiment(
    train_subset_size: int | None,
    test_subset_size: int | None,
    device: torch.device,
    k_neighbors: int,
    images_per_node: int,
) -> tuple[Data, CifarData]:
    """Carga CIFAR-10 y construye el grafo usado por uno o varios modelos."""

    cifar_data = load_cifar10(
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
    )
    graph = _build_graph(
        cifar_data=cifar_data,
        k_neighbors=k_neighbors,
        images_per_node=images_per_node,
    ).to(device)
    return graph, cifar_data


def _build_graph(cifar_data: CifarData, k_neighbors: int, images_per_node: int) -> Data:
    """Extrae features agrupadas y construye el grafo de similitud."""

    train_features, train_labels, train_node_image_counts = build_node_features_from_images(
        images=cifar_data.train.images,
        labels=cifar_data.train.labels,
        images_per_node=images_per_node,
    )
    test_features, test_labels, test_node_image_counts = build_node_features_from_images(
        images=cifar_data.test.images,
        labels=cifar_data.test.labels,
        images_per_node=images_per_node,
    )

    return build_similarity_graph(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        train_node_image_counts=train_node_image_counts,
        test_node_image_counts=test_node_image_counts,
        k_neighbors=k_neighbors,
    )


def _run_model(
    model: nn.Module,
    graph: Data,
    device: torch.device,
    model_type: str,
    dataset_mode: str,
    train_subset_size: int | None,
    test_subset_size: int | None,
    run_mode: str,
    k_neighbors: int,
    epochs: int,
    images_per_node: int,
    run_label: str | None,
) -> dict[str, Any]:
    """Entrena y evalua un modelo durante las epocas configuradas."""

    model = model.to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)
        if trainable_parameters
        else None
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loss = 0.0
    for _ in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            data=graph,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

    train_accuracy = evaluate_accuracy(model, graph, graph.train_mask)
    test_accuracy = evaluate_accuracy(model, graph, graph.test_mask)

    return {
        "run_mode": run_mode,
        "run_label": run_label,
        "model_type": model_type,
        "dataset_mode": dataset_mode,
        "k_neighbors": k_neighbors,
        "epochs": epochs,
        "train_subset_size": train_subset_size
        or int(graph.node_image_counts[graph.train_mask].sum().item()),
        "test_subset_size": test_subset_size
        or int(graph.node_image_counts[graph.test_mask].sum().item()),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_gap": train_accuracy - test_accuracy,
        "inference_time": measure_inference_time(model, graph),
        "num_parameters": count_trainable_parameters(model),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "feature_dim": int(graph.num_node_features),
        "train_nodes": int(graph.train_mask.sum().item()),
        "test_nodes": int(graph.test_mask.sum().item()),
        "total_images": int(graph.node_image_counts.sum().item()),
        "images_per_node": images_per_node,
        "mean_images_per_node": float(graph.node_image_counts.float().mean().item()),
    }


def _print_graph_summary(
    graph: Data,
    cifar_data: CifarData,
    model_name: str,
    dataset_mode: str,
    run_mode: str,
    k_neighbors: int,
    images_per_node: int,
) -> None:
    """Muestra un resumen compacto del grafo construido."""

    y = graph.y.detach().cpu()
    present_class_ids = torch.unique(y).tolist()
    present_classes = [
        f"{class_id}: {cifar_data.class_names[class_id]}" for class_id in present_class_ids
    ]

    print(f"\nResumen del grafo CIFAR-10 para {model_name}")
    print(f"Modo de ejecucion: {run_mode}")
    print(f"Modo de datos: {dataset_mode}")
    print(f"K vecinos: {k_neighbors}")
    print(f"Nodos: {graph.num_nodes}")
    print(f"Aristas: {graph.num_edges}")
    print(f"Dimension de features: {graph.num_node_features}")
    print(f"Nodos train: {int(graph.train_mask.sum().item())}")
    print(f"Nodos test: {int(graph.test_mask.sum().item())}")
    print(f"Imagenes por nodo configuradas: {images_per_node}")
    print(f"Imagenes originales representadas: {int(graph.node_image_counts.sum().item())}")
    print(
        "Media real de imagenes por nodo: "
        f"{graph.node_image_counts.float().mean().item():.2f}"
    )
    print(f"Clases presentes: {', '.join(present_classes)}")


def _print_results_summary(results: dict[str, dict[str, Any]]) -> None:
    """Imprime una comparacion simple de los modelos."""

    print("\nComparacion de modelos")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"Tipo: {metrics['model_type']}")
        print(f"Dataset: {metrics['dataset_mode']}")
        print(f"Train loss: {metrics['train_loss']:.4f}")
        print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Inference time: {metrics['inference_time']:.6f} s")
        print(f"Parametros entrenables: {metrics['num_parameters']}")


if __name__ == "__main__":
    run()
