import pandas as pd
from fpr_experiment.run_experiment import simulate_null_data

X, target = simulate_null_data(N=500, K=10, target_col='fp', binary=True, include=())
target.name = 'performance metric'
X = pd.concat((X, target.astype(int)), axis=1)
X.to_csv('./test.csv')