from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
xlsx = base / 'informe_ejecuciones_tfg.xlsx'
out = base / 'graficas_tfg'
out.mkdir(exist_ok=True)

df = pd.read_excel(xlsx, sheet_name='Resultados')
exp_map = {
    'exp_01_base_100ep': 'Base\n64/16',
    'exp_02_modelos_medios_100ep': 'Medios\n32/8',
    'exp_03_modelos_pequenos_100ep': 'Pequeños\n16/4',
}
model_map = {'mlp_baseline':'MLP', 'full_gnn':'FullGNN', 'tiny_gnn':'TinyGNN', 'tiny_gnn_pinn':'TinyGNN + PINN'}
colors = {'MLP':'#4C78A8','FullGNN':'#F58518','TinyGNN':'#54A24B','TinyGNN + PINN':'#E45756'}
df['Experimento'] = df['experiment_id'].map(exp_map)
df['Modelo'] = df['model'].map(model_map)
order_exp = [exp_map[k] for k in exp_map]
order_model = ['MLP','FullGNN','TinyGNN','TinyGNN + PINN']
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 15, 'axes.labelsize': 12})

def grouped_bar(metric, ylabel, title, filename, log=False):
    fig, ax = plt.subplots(figsize=(10,5.8))
    x = np.arange(len(order_exp)); width = 0.20
    for i,m in enumerate(order_model):
        vals = [float(df[(df.Experimento==e)&(df.Modelo==m)][metric].iloc[0]) for e in order_exp]
        bars = ax.bar(x+(i-1.5)*width, vals, width, label=m, color=colors[m])
        for b,v in zip(bars, vals):
            label = f'{v:.2e}' if v < 0.001 else f'{v:.4f}' if metric != 'num_parameters' else f'{int(v)}'
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), label, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order_exp)
    ax.set_ylabel(ylabel + (' (escala log)' if log else ''))
    ax.set_title(title)
    if log: ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc='upper left')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    p=out/filename; fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    print(p)


grouped_bar('test_mse','Test MSE','Error de predicción por experimento', '01_test_mse.png', log=True)
grouped_bar('test_physics_violation','Violación física','Coherencia física por experimento', '02_violacion_fisica.png', log=True)
grouped_bar('num_parameters','Nº parámetros','Tamaño de cada modelo', '03_parametros.png', log=True)

fig, ax = plt.subplots(figsize=(9,6))
markers = {'MLP':'o','FullGNN':'s','TinyGNN':'^','TinyGNN + PINN':'D'}
for m in order_model:
    sub=df[df.Modelo==m]
    ax.scatter(sub['num_parameters'], sub['test_mse'], s=120, marker=markers[m], color=colors[m], label=m, alpha=0.9)
    for _,r in sub.iterrows():
        ax.annotate(r['Experimento'].replace('\n',' '), (r['num_parameters'], r['test_mse']), textcoords='offset points', xytext=(6,5), fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Nº parámetros (menos = más ligero)')
ax.set_ylabel('Test MSE (menos = mejor)')
ax.set_title('Trade-off: precisión frente a tamaño del modelo')
ax.grid(alpha=0.25, which='both')
ax.legend(frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
p=out/'04_tradeoff_precision_parametros.png'; fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
print(p)
