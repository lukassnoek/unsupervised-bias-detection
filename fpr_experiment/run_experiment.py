from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("error")

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeansV2
from unsupervised_bias_detection.clustering._bahc_v2 import BiasAwareHierarchicalClusteringV2
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


METRICS = {
    'y': lambda y, y_pred: y,
    'y_pred': lambda y, y_pred: y_pred,
    'fp': lambda y, y_pred: np.logical_and(y == 0, y_pred == 1).astype(int),
    'fn': lambda y, y_pred: np.logical_and(y == 1, y_pred == 0).astype(int),
    'err': lambda y, y_pred: (y != y_pred).astype(int),
    'fp_prec': lambda y, y_pred: (y[y_pred == 1] != y_pred[y_pred == 1]).astype(int),
    'fn_rec': lambda y, y_pred: (y[y == 1] != y_pred[y == 1]).astype(int)
}

def simulate_null_data(method, N, K, target_col):

    # Simulate random design matrix `X`
    X = np.random.randn(N * 2, K)
    y = np.random.choice([0, 1], size=N * 2)
    X_train, X, y_train, y = train_test_split(X, y, test_size=0.5, stratify=y)

    model = make_pipeline(StandardScaler(), LogisticRegression())

    # Get cross-validated predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    target = METRICS[target_col](y, y_pred)

    if len(target) == 0:
        # Happens sometimes with fp_prec / fn_rec
        return None
    
    # Obtain cluster labels
    if method == 'hbac':
        hbac = BiasAwareHierarchicalKMeansV2(n_iter=10, min_cluster_size=5)
        hbac.fit(X, target)
        labels = hbac.labels_
        return X, target, labels    
    elif method == 'randomclusters':
        labels = np.random.choice(range(5), size=X.shape[0])
        return X, target, labels
    else:
        raise ValueError(f"Not a known method ({method})")
    

def check_before_test(c0, c1, min_samples=5):
    """Determines whether to do significance testing
    (don't do this with too few observations/constant data to avoid NaNs)."""
    if (c0.size < min_samples) or (c1.size < min_samples):
        return True

    if np.mean(c0) == np.mean(c1):
        return True
    
    if np.all(c0 == c0[0]) or np.all(c1 == c1[0]):
        return True

    return False


def simulate_experiment(method, N, K, target_col, bonf_correct=False, iters=100):
    
    results_clust = defaultdict(list)
    results_feat = defaultdict(list)

    params_ = list(zip(
        ['method', 'N', 'K', 'target_col', 'bonf_correct'],
        [method, N, K, target_col, bonf_correct]
    ))

    for i in range(iters):

        out = simulate_null_data(method, N, K, target_col)

        if out is None:
            # Sometimes the data is empty because there are no FPs/FNs
            continue

        X, target, labels = out
        if X.shape[0] <= 5:
            # Sometimes there are too few observations
            continue

        # For each cluster, compute statistics separately
        n_clust = np.unique(labels).size
        for  l in np.unique(labels):
            # Define cluster 1 (with label) and cluster 0 (~label)
            idx = labels == l
            c1, c0 = target[idx], target[~idx]
            should_continue = check_before_test(c0, c1)
            if should_continue:
                # With too few data/constant data, stats cannot be computed, so continue
                continue

            # Compute cluster stats and save in container
            c1, c0 = target[idx], target[~idx]
            _, p_clust = ttest_ind(c1, c0, equal_var=False)

            if bonf_correct:
                p_clust = p_clust * n_clust

            diff_clust = c1.mean() - c0.mean()
    
            results_clust['iter'].append(i)
            results_clust['cluster_nr'].append(l)
            results_clust['p_clust'].append(p_clust)
            results_clust['diff_clust'].append(diff_clust)
            results_clust['size_clust'].append(idx.sum())

            for p_name_, p_ in params_:
                # Save params
                results_clust[p_name_].append(p_)

    results_clust = pd.DataFrame(results_clust)
    
    return results_clust


if __name__ == '__main__':

    from params import ITERS
    from params import params

    results_clust = []
    for params in tqdm(params):

        method, n, k, target_col, bonf_correct = params

        results_clust_ = simulate_experiment(
            # Run experiment with `method`
            method, n, k, target_col, bonf_correct=bonf_correct, iters=ITERS
        )
        results_clust.append(results_clust_)

    results_clust = pd.concat(results_clust, axis=0)
    f_out = Path(__file__).parent / 'results_clust.csv'
    results_clust.to_csv(f_out, index=False)
