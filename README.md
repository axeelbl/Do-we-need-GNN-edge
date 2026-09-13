# Do we need GNN at the edge?

Reproducible experiments comparing MLP, Full-GNN, Tiny-GNN and Tiny-GNN + PINN models for thermal prediction on a grid graph.

Este repositorio contiene el código y los resultados finales usados para la memoria del TFG, comparando MLP, Full-GNN, Tiny-GNN y Tiny-GNN + PINN en predicción térmica sobre una cuadrícula/grafo.

## Estructura de entrega

- `src/`: código activo de simulación, modelos, entrenamiento y sweep corregido.
- `CONCLUSIONES/`: resultados finales limpios para la memoria/entrega.
  - `Results.csv`: CSV definitivo usado para la memoria (`2970` filas), con columnas de hardware/GPU y metadatos de ejecución.
  - `Report.xlsx`: Excel final regenerado desde los resultados definitivos, sin filas `smoke`.
  - `Informe_analisis_experimentos.docx`: informe final coherente con `Results.csv` y `Report.xlsx`.
  - `figuras/`: figuras usadas por el informe.
  - `tablas/`: tablas derivadas del CSV final.
- `requirements.txt`: dependencias mínimas para ejecutar el código.

Las carpetas y resultados antiguos no se incluyen en la entrega final para evitar confusiones con versiones previas del análisis.

## Experimento principal

El sweep corregido está en:

```bash
python src/run_sweep_experiments_v2.py --mode recommended
```

Los resultados finales comparables usan:

- seeds `40..50`
- checkpoints `50, 100, 300`
- hidden dimensions `4, 8, 16, 32`
- data fractions `0.125, 0.25, 0.5, 1.0`
- noise levels `0.0, 0.05, 0.10, 0.20`
- PINN lambdas `0.001, 0.01, 0.1, 1, 10, 30`

Para Tiny-GNN + PINN, `lambda` se selecciona por `validation MSE` por configuración; no se selecciona usando test.

## Resumen de resultados

- Full-GNN obtiene el menor MSE medio global.
- MLP gana más configuraciones individuales, especialmente en escenarios limpios/simples.
- Tiny-GNN es la opción ligera: muchos menos parámetros, pero peor precisión global.
- Tiny-GNN + PINN mejora a Tiny-GNN de forma clara, especialmente con ruido.

## Reproducibilidad

Se requiere Python 3.11 o posterior. Para crear un entorno limpio e instalar las dependencias de ejecución:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El código selecciona CUDA automáticamente si está disponible. Puede forzarse un dispositivo, por ejemplo:

```bash
TFG_DEVICE=cpu python src/run_sweep_experiments_v2.py --mode smoke
```

Los resultados nuevos se escriben por defecto en `src/results/config_sweep/corrected_experiments/`, una ruta ignorada por Git. Use `--output-dir` para elegir otra ubicación. El modo `smoke` es una validación rápida; `recommended` y `full` requieren bastante más tiempo y recursos.

## Calidad y seguridad

Instale también las herramientas de desarrollo y ejecute las mismas comprobaciones que CI:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q src tests
ruff check --select E4,E7,E9,F --ignore E402 src tests
python -m unittest discover -s tests -v
python src/run_sweep_experiments_v2.py --mode smoke --output-dir /tmp/gnn-smoke
pip-audit -r requirements.txt
pip-audit
```

La segunda auditoría incluye tanto las dependencias de ejecución como las herramientas de desarrollo instaladas. GitHub Actions ejecuta estas comprobaciones en cada cambio dirigido a `main`.

## Docker

La imagen ejecuta el experimento `smoke` como usuario sin privilegios:

```bash
docker build -t do-we-need-gnn-edge .
docker run --rm do-we-need-gnn-edge
```

Para conservar resultados, monte una carpeta escribible y seleccione explícitamente la salida:

```bash
docker run --rm -v "$PWD/results:/results" do-we-need-gnn-edge \
  python src/run_sweep_experiments_v2.py --mode smoke --output-dir /results
```

## Alcance de los resultados

Los artefactos de `CONCLUSIONES/` son resultados de investigación ya generados, no una garantía de rendimiento en otro hardware. Las métricas de inferencia dependen especialmente del dispositivo indicado en cada fila; para comparaciones nuevas, regenere los experimentos en un entorno común.
