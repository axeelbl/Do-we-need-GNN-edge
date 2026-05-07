"""Pruebas del pipeline de datos con tensores pequenos."""

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.feature_extractor import build_node_features_from_images, extract_features
from data.graph_builder import build_similarity_graph


class DataPipelineTests(unittest.TestCase):
    """Comprueba extraccion de features y construccion del grafo."""

    def test_extract_features_flattens_images(self) -> None:
        images = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

        features = extract_features(images, normalize=False)

        self.assertEqual(features.shape, (2, 12))
        self.assertTrue(torch.equal(features[0], images[0].reshape(-1)))

    def test_build_node_features_groups_images_by_class(self) -> None:
        images = torch.arange(6 * 3 * 2 * 2, dtype=torch.float32).reshape(6, 3, 2, 2)
        labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        features, node_labels, image_counts = build_node_features_from_images(
            images=images,
            labels=labels,
            images_per_node=2,
            normalize=False,
        )

        self.assertEqual(features.shape, (4, 12))
        self.assertEqual(node_labels.tolist(), [0, 0, 1, 1])
        self.assertEqual(image_counts.tolist(), [2, 1, 2, 1])

    def test_build_similarity_graph_creates_masks_and_counts(self) -> None:
        train_features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        test_features = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        train_labels = torch.tensor([0, 1, 0])
        test_labels = torch.tensor([1, 1])
        train_counts = torch.tensor([2, 2, 1])
        test_counts = torch.tensor([2, 1])

        graph = build_similarity_graph(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            train_node_image_counts=train_counts,
            test_node_image_counts=test_counts,
            k_neighbors=1,
        )

        self.assertEqual(graph.x.shape, (5, 2))
        self.assertEqual(graph.edge_index.shape[0], 2)
        self.assertEqual(int(graph.train_mask.sum()), 3)
        self.assertEqual(int(graph.test_mask.sum()), 2)
        self.assertEqual(graph.node_image_counts.tolist(), [2, 2, 1, 2, 1])


if __name__ == "__main__":
    unittest.main()
