# TFG edge cooling: resultados finales

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

Instalar dependencias:

```bash
pip install -r requirements.txt
```

El código selecciona CUDA automáticamente si está disponible; se puede forzar con `TFG_DEVICE`.
