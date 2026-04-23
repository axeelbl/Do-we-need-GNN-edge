from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from config import DEVICE, K_NEIGHBORS, LEARNING_RATE, RANDOM_SEED, RESULTS_DIR
from data.cifar_loader import CifarData, load_cifar10
from data.feature_extractor import extract_features
from data.graph_builder import build_similarity_graph
from models.full_gnn import FullGNN
from models.tiny_gnn import TinyGNN
from training.evaluator import evaluate_accuracy, measure_inference_time
from training.trainer import train_one_epoch
from utils.metrics import count_trainable_parameters, save_metrics
from utils.seed import set_seed


def run() -> None:
    """Ejecuta una comparacion entre Full-GNN y Tiny-GNN."""

    # Fijamos semilla para repetir el mismo experimento.
    set_seed(RANDOM_SEED)

    # Dispositivo donde se ejecuta el modelo: CPU por defecto ahora mismo
    device = torch.device(DEVICE)

    # Cargamos CIFAR-10 y construimos el grafo de similitud.
    cifar_data = load_cifar10()
    graph = _build_graph(cifar_data).to(device)

    # Mostramos informacion basica del grafo antes de entrenar.
    _print_graph_summary(graph, cifar_data)

    # Instanciamos los dos modelos que se van a comparar.
    models = {
        "full_gnn": FullGNN(),
        "tiny_gnn": TinyGNN(),
    }

    # Guardaremos las metricas de cada modelo 
    results: dict[str, dict[str, float | int]] = {}

    # Entrenamos y evaluamos cada modelo una sola vez.
    for model_name, model in models.items():
        results[model_name] = _run_model(model, graph, device)

    # Persistimos las metricas para poder revisarlas despues.
    metrics_path = RESULTS_DIR / "metrics.json"
    save_metrics(metrics_path, results)

    # Imprimimos una comparacion sencilla por consola.
    _print_results_summary(results)
    print(f"Metricas guardadas en: {metrics_path}")


def _build_graph(cifar_data: CifarData) -> Data:
    """Extrae features y construye el grafo de similitud."""

    # Convertimos imagenes de train a vectores numericos.
    train_features = extract_features(cifar_data.train.images)

    # Convertimos imagenes de test usando el mismo extractor.
    test_features = extract_features(cifar_data.test.images)

    # Construimos un unico grafo con nodos de train y test.
    return build_similarity_graph(
        train_features=train_features,
        train_labels=cifar_data.train.labels,
        test_features=test_features,
        test_labels=cifar_data.test.labels,
        k_neighbors=K_NEIGHBORS,
    )


def _run_model(
    model: nn.Module,
    graph: Data,
    device: torch.device,
) -> dict[str, float | int]:
    """Entrena y evalua un modelo"""

    # Movemos el modelo al mismo dispositivo que el grafo.
    model = model.to(device)

    # Adam es un optimizador simple y habitual para esta primera prueba.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CrossEntropyLoss sirve para clasificacion multiclase con logits.
    loss_fn = nn.CrossEntropyLoss()

    # Entrenamos exactamente una epoca.
    train_loss = train_one_epoch(
        model=model,
        data=graph,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    # Calculamos metricas basicas despues de entrenar.
    return {
        "train_loss": train_loss,
        "train_accuracy": evaluate_accuracy(model, graph, graph.train_mask),
        "test_accuracy": evaluate_accuracy(model, graph, graph.test_mask),
        "inference_time": measure_inference_time(model, graph),
        "num_parameters": count_trainable_parameters(model),
    }


def _print_graph_summary(graph: Data, cifar_data: CifarData) -> None:
    """Muestra un resumen compacto del grafo construido."""

    # Pasamos las etiquetas a CPU para imprimirlas de forma segura.
    y = graph.y.detach().cpu()

    # Obtenemos las clases que aparecen realmente en el subconjunto usado.
    present_class_ids = torch.unique(y).tolist()
    present_classes = [
        f"{class_id}: {cifar_data.class_names[class_id]}" for class_id in present_class_ids
    ]

    # Resumen util para comprobar que el grafo se ha creado bien.
    print("Resumen del grafo CIFAR-10")
    print(f"Nodos: {graph.num_nodes}")
    print(f"Aristas: {graph.num_edges}")
    print(f"Dimension de features: {graph.num_node_features}")
    print(f"Nodos train: {int(graph.train_mask.sum().item())}")
    print(f"Nodos test: {int(graph.test_mask.sum().item())}")
    print(f"Clases presentes: {', '.join(present_classes)}")


def _print_results_summary(results: dict[str, dict[str, Any]]) -> None:
    """Imprime una comparacion simple de los modelos."""

    # Recorremos las metricas guardadas para cada modelo.
    print("\nComparacion de modelos")
    for model_name, metrics in results.items():
        # Formateamos los numeros para que la consola sea facil de leer.
        print(f"\n{model_name}")
        print(f"Train loss: {metrics['train_loss']:.4f}")
        print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Inference time: {metrics['inference_time']:.6f} s")
        print(f"Parametros entrenables: {metrics['num_parameters']}")


if __name__ == "__main__":
    # Permite ejecutar el pipeline con: python src/main.py
    run()
