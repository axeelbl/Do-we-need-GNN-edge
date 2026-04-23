# TFG GNN

Base inicial para un TFG experimental en Python orientado a comparar dos
modelos GNN sobre un problema estructurado a partir de CIFAR-10:

- **Full-GNN**: modelo principal con mayor capacidad.
- **Tiny-GNN**: modelo reducido para comparar coste y rendimiento.

Esta primera fase no implementa entrenamiento completo, task offloading ni
logica avanzada. El objetivo es dejar una estructura limpia y preparada para
crecer.

## Estructura

```text
tfg_gnn/
|-- data/
|   `-- raw/
|-- src/
|   |-- config.py
|   |-- main.py
|   |-- data/
|   |   |-- cifar_loader.py
|   |   |-- feature_extractor.py
|   |   `-- graph_builder.py
|   |-- models/
|   |   |-- full_gnn.py
|   |   `-- tiny_gnn.py
|   |-- training/
|   |   |-- trainer.py
|   |   `-- evaluator.py
|   |-- utils/
|   |   |-- seed.py
|   |   |-- metrics.py
|   |   `-- io.py
|   `-- results/
|       `-- metrics.json
|-- requirements.txt
`-- README.md
```

## Uso inicial

Desde la carpeta del proyecto:

```bash
python -m venv .venv
pip install -r requirements.txt
python src/main.py
```

`main.py` inicializa la semilla, asegura los directorios necesarios y escribe
un fichero `src/results/metrics.json` con el estado inicial del experimento.

## Siguiente fase

El siguiente paso natural es implementar `src/data/cifar_loader.py` con
`torchvision.datasets.CIFAR10`, transformaciones basicas y loaders de train/test.
Despues se podra anadir un extractor de caracteristicas simple y construir el
primer grafo con vecinos cercanos.
