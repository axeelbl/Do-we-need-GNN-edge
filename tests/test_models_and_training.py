"""Pruebas de modelos, entrenamiento y evaluacion con un grafo sintetico."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.baselines import RandomBaseline, SimpleMLPBaseline
from models.tiny_gnn import TinyGNN
from training.evaluator import evaluate_accuracy, measure_inference_time
from training.trainer import model_forward, train_one_epoch
from utils.metrics import count_trainable_parameters


class ModelTrainingTests(unittest.TestCase):
    """Comprueba forward, entrenamiento minimo y metricas basicas."""

    def setUp(self) -> None:
        self.graph = Data(
            x=torch.randn(6, 4),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 3, 4, 4, 5],
                    [1, 0, 2, 1, 4, 3, 5, 4],
                ],
                dtype=torch.long,
            ),
            y=torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long),
            train_mask=torch.tensor([True, True, True, False, False, False]),
            test_mask=torch.tensor([False, False, False, True, True, True]),
            node_image_counts=torch.ones(6, dtype=torch.long),
        )

    def test_model_forward_supports_graph_and_no_graph_models(self) -> None:
        tiny_gnn = TinyGNN(input_dim=4, hidden_dim=5, num_classes=3)
        mlp = SimpleMLPBaseline(input_dim=4, hidden_dim=5, num_classes=3)

        self.assertEqual(model_forward(tiny_gnn, self.graph).shape, (6, 3))
        self.assertEqual(model_forward(mlp, self.graph).shape, (6, 3))

    def test_train_one_epoch_updates_trainable_model(self) -> None:
        model = SimpleMLPBaseline(input_dim=4, hidden_dim=5, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        loss = train_one_epoch(model, self.graph, optimizer, loss_fn, device="cpu")

        self.assertGreaterEqual(loss, 0.0)

    def test_random_baseline_runs_without_optimizer(self) -> None:
        model = RandomBaseline(num_classes=3)
        loss_fn = nn.CrossEntropyLoss()

        loss = train_one_epoch(model, self.graph, optimizer=None, loss_fn=loss_fn, device="cpu")
        accuracy = evaluate_accuracy(model, self.graph, self.graph.test_mask)

        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertEqual(count_trainable_parameters(model), 0)

    def test_measure_inference_time_returns_seconds(self) -> None:
        model = SimpleMLPBaseline(input_dim=4, hidden_dim=5, num_classes=3)

        inference_time = measure_inference_time(model, self.graph)

        self.assertGreaterEqual(inference_time, 0.0)


if __name__ == "__main__":
    unittest.main()
