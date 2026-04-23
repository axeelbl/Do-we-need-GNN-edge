"""Carga de CIFAR-10 y seleccion de subconjuntos."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from config import DATA_DIR, RANDOM_SEED, TEST_SUBSET_SIZE, TRAIN_SUBSET_SIZE


@dataclass(frozen=True)
class CifarSplit:
    """Imagenes y etiquetas de un split de CIFAR-10."""

    # Tensor con forma (num_imagenes, canales, alto, ancho)
    images: torch.Tensor

    # Etiqueta numerica de cada imagen
    labels: torch.Tensor


@dataclass(frozen=True)
class CifarData:
    """Datos de train/test preparados para el pipeline."""

    # Subconjunto de entrenamiento
    train: CifarSplit

    # Subconjunto de test
    test: CifarSplit

    # Nombres de las 10 clases de CIFAR-10
    class_names: tuple[str, ...]


def get_cifar10_class_names() -> tuple[str, ...]:
    """Devuelve los nombres oficiales de clases de CIFAR-10."""

    # Orden oficial que usa torchvision para CIFAR-10.
    return (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )


def load_cifar10(
    data_dir: Path = DATA_DIR,
    train_subset_size: int = TRAIN_SUBSET_SIZE,
    test_subset_size: int = TEST_SUBSET_SIZE,
    seed: int = RANDOM_SEED,
    download: bool = True,
) -> CifarData:
    """Descarga/carga CIFAR-10 y devuelve subconjuntos como tensores."""

    # ToTensor convierte la imagen PIL a tensor con valores entre 0 y 1.
    transform = transforms.Compose([transforms.ToTensor()])

    # Dataset completo de entrenamiento.
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )

    # Dataset completo de test.
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    # Elegimos un subconjunto reproducible de train.
    train_indices = _select_subset_indices(
        dataset_size=len(train_dataset),
        subset_size=train_subset_size,
        seed=seed,
    )

    # Elegimos otro subconjunto reproducible de test.
    test_indices = _select_subset_indices(
        dataset_size=len(test_dataset),
        subset_size=test_subset_size,
        seed=seed + 1,
    )

    # Convertimos los dos subconjuntos a tensores para el resto del pipeline.
    return CifarData(
        train=_subset_to_tensors(train_dataset, train_indices),
        test=_subset_to_tensors(test_dataset, test_indices),
        class_names=tuple(train_dataset.classes),
    )


def _select_subset_indices(
    dataset_size: int,
    subset_size: int,
    seed: int,
) -> list[int]:
    """Selecciona indices aleatorios reproducibles para un subconjunto."""

    # Evita subconjuntos vacios o negativos.
    if subset_size <= 0:
        raise ValueError("subset_size debe ser mayor que 0.")

    # Evita pedir mas muestras de las que existen.
    if subset_size > dataset_size:
        raise ValueError(
            f"subset_size={subset_size} supera el tamano del dataset ({dataset_size})."
        )

    # randperm genera una permutacion aleatoria usando la semilla indicada.
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator)[:subset_size].tolist()


def _subset_to_tensors(dataset: datasets.CIFAR10, indices: list[int]) -> CifarSplit:
    """Convierte un subconjunto de CIFAR-10 a tensores."""

    # Subset permite leer solo los indices seleccionados.
    subset = Subset(dataset, indices)
    images: list[torch.Tensor] = []
    labels: list[int] = []

    # Recorremos el subconjunto y acumulamos imagenes y etiquetas.
    for image, label in subset:
        images.append(image)
        labels.append(label)

    # Stack junta todas las imagenes en un unico tensor.
    return CifarSplit(
        images=torch.stack(images),
        labels=torch.tensor(labels, dtype=torch.long),
    )
