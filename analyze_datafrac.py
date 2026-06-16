import pandas as pd
from pathlib import Path
p=Path('src/results/data_fraction_gpu/data_fraction_gpu_runs.csv')
df=pd.read_csv(p)
agg=df.groupby(['data_fraction','model_label'], as_index=False).agg(
    test_mse_mean=('test_mse','mean'),
    test_mse_std=('test_mse','std'),
    phys_mean=('test_physics_violation','mean'),
    params=('num_parameters','mean'),
    runtime_mean=('runtime_seconds','mean'),
    n=('seed','nunique'),
).sort_values(['data_fraction','test_mse_mean'], ascending=[False, True])
print('BEST_BY_FRACTION')
print(agg.groupby('data_fraction').first()[['model_label','test_mse_mean','test_mse_std','params']].to_string())
print('\nPIVOT_MSE_MEAN')
print(agg.pivot(index='data_fraction', columns='model_label', values='test_mse_mean').sort_index(ascending=False).round(6).to_string())
print('\nPIVOT_MSE_STD')
print(agg.pivot(index='data_fraction', columns='model_label', values='test_mse_std').sort_index(ascending=False).round(6).to_string())
print('\nTRAIN_PAIRS')
print(df.groupby('data_fraction')['train_pairs'].first().sort_index(ascending=False).to_string())
print('\nFILES')
print(p.resolve())
print(Path('informe_data_fraction_gpu.xlsx').resolve())
