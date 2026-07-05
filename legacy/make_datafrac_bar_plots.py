from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path.cwd()
results = root / 'src' / 'results' / 'data_fraction_gpu'
out = root / 'graficas_tfg' / 'data_fraction_gpu_barras'
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(results / 'data_fraction_gpu_runs.csv')

model_order = ['MLP', 'FullGNN', 'TinyGNN', 'TinyGNN + PINN']
colors = {'MLP':'#4C78A8','FullGNN':'#F58518','TinyGNN':'#54A24B','TinyGNN + PINN':'#E45756'}
fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
labels = ['100%', '50%', '20%', '10%', '5%', '1%']

agg = df.groupby(['data_fraction','model_label'], as_index=False).agg(
    test_mse_mean=('test_mse','mean'),
    test_mse_std=('test_mse','std'),
    phys_mean=('test_physics_violation','mean'),
    phys_std=('test_physics_violation','std'),
    infer_mean=('inference_time','mean'),
    infer_std=('inference_time','std'),
    params=('num_parameters','mean'),
    n=('seed','nunique'),
)

def grouped_bar(metric, err, ylabel, title, filename, logy=False):
    x = np.arange(len(fractions))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(model_order):
        vals, errs = [], []
        for f in fractions:
            row = agg[(agg.data_fraction == f) & (agg.model_label == model)]
            vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row[err].iloc[0]) if err and not row.empty else 0.0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, yerr=errs if err else None, capsize=3, label=model, color=colors[model], alpha=0.92)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Porcentaje de datos de entrenamiento')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if logy:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)

def best_bar(filename):
    sub = agg.sort_values(['data_fraction','test_mse_mean']).groupby('data_fraction', as_index=False).first()
    sub['label'] = sub['data_fraction'].map(dict(zip(fractions, labels)))
    sub = sub.set_index('data_fraction').loc[fractions].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_colors = [colors.get(m, '#888') for m in sub.model_label]
    bars = ax.bar(sub.label, sub.test_mse_mean, color=bar_colors, alpha=0.92)
    ax.set_title('Mejor modelo por porcentaje de datos', fontsize=14, weight='bold')
    ax.set_xlabel('Porcentaje de datos de entrenamiento')
    ax.set_ylabel('Mejor Test MSE medio')
    ax.grid(axis='y', alpha=0.25)
    for bar, model, val in zip(bars, sub.model_label, sub.test_mse_mean):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{model}\n{val:.4g}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)

def single_fraction_bar(frac, filename):
    sub = agg[agg.data_fraction == frac].set_index('model_label').loc[model_order].reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(sub.model_label, sub.test_mse_mean, yerr=sub.test_mse_std, capsize=4, color=[colors[m] for m in sub.model_label], alpha=0.92)
    ax.set_title(f'Comparación de modelos con {int(frac*100)}% de datos', fontsize=14, weight='bold')
    ax.set_ylabel('Test MSE medio ± std')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, sub.test_mse_mean):
        ax.text(bar.get_x()+bar.get_width()/2, val, f'{val:.4g}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)

# Gráficas principales tipo barras
grouped_bar('test_mse_mean', 'test_mse_std', 'Test MSE medio ± std', 'Test MSE por cantidad de datos', '01_barras_test_mse.png')
grouped_bar('test_mse_mean', 'test_mse_std', 'Test MSE medio ± std (log)', 'Test MSE por cantidad de datos (escala log)', '02_barras_test_mse_log.png', logy=True)
grouped_bar('phys_mean', 'phys_std', 'Violación física media ± std', 'Violación física por cantidad de datos', '03_barras_violacion_fisica.png')
grouped_bar('infer_mean', 'infer_std', 'Tiempo de inferencia medio (s)', 'Tiempo de inferencia por modelo', '04_barras_tiempo_inferencia.png')
best_bar('05_barras_mejor_modelo_por_fraction.png')
single_fraction_bar(1.0, '06_barras_modelos_100_datos.png')
single_fraction_bar(0.01, '07_barras_modelos_1_datos.png')

print(out)
for p in sorted(out.glob('*.png')):
    print(p)
