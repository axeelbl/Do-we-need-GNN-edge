# TFG GNN

Proyecto experimental en Python para comparar modelos basados en grafos sobre
CIFAR-10. Las imagenes se transforman en vectores, se agrupan en nodos y se
conectan mediante KNN para evaluar si una GNN pequena puede acercarse al
rendimiento de una GNN de mayor capacidad usando menos datos y menos recursos.

El pipeline compara cuatro enfoques:

- `RandomBaseline`: prediccion aleatoria como referencia minima.
- `SimpleMLPBaseline`: clasificador sin grafo sobre las features de los nodos.
- `TinyGNN`: GCN reducida para el escenario ligero.
- `FullGNN`: GCN con mayor capacidad para el escenario completo.

## Estructura

```text
tfg_gnn/
|-- .github/workflows/tests.yml
|-- data/
|   `-- raw/
|-- scripts/
|   `-- run_config_sweep.py
|-- src/
|   |-- config.py
|   |-- main.py
|   |-- data/
|   |-- models/
|   |-- training/
|   |-- utils/
|   `-- results/
|-- tests/
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Uso

Desde la carpeta del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

La primera ejecucion descarga CIFAR-10 en `data/raw/`. El pipeline guarda las
metricas en `src/results/metrics.json` y `src/results/metrics.csv`, y genera
graficos comparativos en `src/results/plots/`.

## Modos de Experimento

El modo principal esta definido en `src/config.py`:

```python
RUN_MODE = "resource_efficiency"
```

En este modo, `FullGNN` usa todo CIFAR-10, mientras que `TinyGNN` y los
baselines trabajan con un subconjunto. El objetivo es comparar rendimiento,
tiempo de inferencia y numero de parametros en un escenario de eficiencia de
recursos.

Tambien se puede usar:

```python
RUN_MODE = "controlled_subset"
```

En este segundo modo todos los modelos usan el mismo subconjunto de datos. Sirve
como control para separar el efecto del modelo del efecto del tamano del dataset.

## Barrido de Configuraciones

El script `scripts/run_config_sweep.py` ejecuta varias combinaciones de
`k_neighbors`, `epochs` e `images_per_node`:

```bash
python scripts/run_config_sweep.py
```

Los valores por defecto se definen en `src/config.py`:

```python
SWEEP_K_NEIGHBORS = [5, 8]
SWEEP_EPOCHS = [1, 3]
SWEEP_IMAGES_PER_NODE = [1, 5, 10]
```

Tambien se pueden pasar valores por consola:

```bash
python scripts/run_config_sweep.py --k-neighbors 5 8 --epochs 1 3 --images-per-node 1 5
```

El resumen global se guarda en:

```text
src/results/config_sweep/sweep_metrics.csv
src/results/config_sweep/sweep_metrics.json
```

## Docker y Tests

Construir la imagen:

```bash
docker build -t tfg-gnn .
```

Ejecutar el barrido dentro del contenedor:

```bash
docker run --rm -v ${PWD}\data\raw:/app/data/raw -v ${PWD}\src\results:/app/src/results tfg-gnn
```

Ejecutar el pipeline principal:

```bash
docker run --rm tfg-gnn python src/main.py
```

Ejecutar las pruebas:

```bash
python -m unittest discover -s tests
```

O dentro de Docker:

```bash
docker run --rm tfg-gnn python -m unittest discover -s tests
```
