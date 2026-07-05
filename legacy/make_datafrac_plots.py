from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path.cwd()
results = root / 'src' / 'results' / 'data_fraction_gpu'
out = root / 'graficas_tfg' / 'data_fraction_gpu'
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(results / 'data_fraction_gpu_runs.csv')

order = ['MLP','FullGNN','TinyGNN','TinyGNN + PINN']
colors = {'MLP':'#4C78A8','FullGNN':'#F58518','TinyGNN':'#54A24B','TinyGNN + PINN':'#E45756'}
agg = df.groupby(['data_fraction','model_label'], as_index=False).agg(
    test_mse_mean=('test_mse','mean'), test_mse_std=('test_mse','std'),
    phys_mean=('test_physics_violation','mean'), phys_std=('test_physics_violation','std'),
    infer_mean=('inference_time','mean'), infer_std=('inference_time','std'),
    params=('num_parameters','mean'), n=('seed','nunique')
)

def line(metric_mean, metric_std, ylabel, title, filename, logy=False):
    fig, ax = plt.subplots(figsize=(9,5.5))
    for model in order:
        sub = agg[agg.model_label == model].sort_values('data_fraction')
        if sub.empty: continue
        ax.errorbar(sub.data_fraction, sub[metric_mean], yerr=sub[metric_std], marker='o', capsize=3, label=model, color=colors[model])
    ax.set_title(title)
    ax.set_xlabel('data_fraction')
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()
    ax.grid(True, alpha=.3)
    if logy: ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)

line('test_mse_mean','test_mse_std','Test MSE (media ± std)','Data fraction vs Test MSE','01_data_fraction_test_mse.png')
line('test_mse_mean','test_mse_std','Test MSE log (media ± std)','Data fraction vs Test MSE (escala log)','02_data_fraction_test_mse_log.png', logy=True)
line('phys_mean','phys_std','Violación física (media ± std)','Data fraction vs violación física','03_data_fraction_physics_violation.png')

base = agg[agg.data_fraction == 1.0][['model_label','test_mse_mean']].rename(columns={'test_mse_mean':'base_mse'})
rel = agg.merge(base, on='model_label')
rel['relative_mse'] = rel['test_mse_mean'] / rel['base_mse']
fig, ax = plt.subplots(figsize=(9,5.5))
for model in order:
    sub = rel[rel.model_label == model].sort_values('data_fraction')
    if sub.empty: continue
    ax.plot(sub.data_fraction, sub.relative_mse, marker='o', label=model, color=colors[model])
ax.axhline(1.0, color='black', linewidth=1, alpha=.5)
ax.set_title('Degradación relativa respecto a data_fraction=1.0')
ax.set_xlabel('data_fraction')
ax.set_ylabel('Test MSE relativo')
ax.invert_xaxis()
ax.grid(True, alpha=.3)
ax.legend()
fig.tight_layout()
fig.savefig(out / '04_data_fraction_relative_degradation.png', dpi=180)
plt.close(fig)

for frac in [1.0, 0.01]:
    sub = agg[agg.data_fraction == frac]
    fig, ax = plt.subplots(figsize=(8,5))
    for _, r in sub.iterrows():
        ax.scatter(r.params, r.test_mse_mean, s=90, color=colors.get(r.model_label, 'gray'))
        ax.annotate(r.model_label, (r.params, r.test_mse_mean), xytext=(6,4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Nº parámetros (log)')
    ax.set_ylabel('Test MSE medio (log)')
    ax.set_title(f'Precisión vs tamaño del modelo (data_fraction={frac})')
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(out / f'05_params_vs_mse_df{str(frac).replace(".","p")}.png', dpi=180)
    plt.close(fig)

print(out)
for p in sorted(out.glob('*.png')):
    print(p)
