from copy import copy
from itertools import product
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
from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeans
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def simulate_null_data(method, N, K, target_col):

    # This ensures a perfectly balanced binary target 
    y = np.random.permutation([0] * N + [1] * N)
    model = make_pipeline(StandardScaler(), LogisticRegression())

    # Simulate random design matrix `X`
    X = np.random.randn(N*2, K)

    # Split in train and test set
    X_train, X, y_train, y = train_test_split(X, y, test_size=0.5, stratify=y)

    # Obtain cross-validated predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X)

    # Define target
    if target_col == 'y':
        target = y
    elif target_col == 'y_pred':
        target = y_pred    
    elif target_col == 'fp':
        target = np.logical_and(y == 0, y_pred == 1).astype(int)
    elif target_col == 'fn':
        target = np.logical_and(y == 1, y_pred == 0).astype(int)
    elif target_col == 'err':
        target = (y != y_pred).astype(int)
    elif target_col == 'fp_prec':
        target = (y[y_pred == 1] != y_pred[y_pred == 1]).astype(int)
        X = X[y_pred == 1, :]
    elif target_col == 'fn_rec':
        target = (y[y == 1] != y_pred[y == 1]).astype(int)
        X = X[y == 1, :]
    else:
        raise ValueError("Unknown target col")
    
    if len(target) == 0:
        # Happens sometimes with fp_prec / fn_rec
        return None
    
    # Obtain cluster labels
    if method in ['kmeans', 'kmeans_cv']:
        cluster_model = KMeans(n_clusters=5)
        if method == 'kmeans':
            cluster_model.fit(X_train)
            labels = cluster_model.predict(X)
        else:
            labels = cluster_model.fit_predict(X)
    elif method == 'hbac':
        hbac = BiasAwareHierarchicalKMeans(n_iter=10, min_cluster_size=5)
        hbac.fit(X, target)
        labels = hbac.labels_
    elif method == 'randomclusters':
        labels = np.random.choice(range(5), size=X.shape[0])
    else:
        raise ValueError(f"Not a known method ({method})")

    return X, target, labels


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


def compute_statistics(X, target, idx, bonf_correct, n_clust):
    """Compute statistics for each cluster/feature."""
    c1, c0 = target[idx], target[~idx]
    _, p_clust = ttest_ind(c1, c0, equal_var=False)
    
    if bonf_correct:
        p_clust = p_clust * n_clust

    diff_clust = c1.mean() - c0.mean()
    
    p_feat, diff_feat = [], []
    for ii in range(X.shape[1]):
        X_ = X[:, ii]
        c1, c0 = X_[idx], X_[~idx]

        should_continue = check_before_test(c0, c1)
        if should_continue:
            continue
        
        _, p = ttest_ind(c1, c0)
        if bonf_correct:
            p = p * X.shape[1]

        p_feat.append(p)
        diff = c1.mean() - c0.mean()
        diff_feat.append(diff)

    return p_clust, diff_clust, p_feat, diff_feat


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
            p_clust, diff_clust, p_feat, diff_feat = compute_statistics(X, target, idx, bonf_correct, n_clust)
            results_clust['iter'].append(i)
            results_clust['cluster_nr'].append(l)
            results_clust['p_clust'].append(p_clust)
            results_clust['diff_clust'].append(diff_clust)
            results_clust['size_clust'].append(idx.sum())

            for p_name_, p_ in params_:
                # Save params
                results_clust[p_name_].append(p_)

            # Save feature stats separately
            for i_feat, (p_feat_, diff_feat_) in enumerate(zip(p_feat, diff_feat)):
                results_feat['iter'].append(i)
                results_feat['feat_nr'].append(i_feat)
                results_feat['p_feat'].append(p_feat_)
                results_feat['diff_feat'].append(diff_feat_)
                results_feat['size_feat'].append(idx.sum())

                for p_name, p_ in params_:
                    results_feat[p_name].append(p_)

    results_clust = pd.DataFrame(results_clust)
    results_feat = pd.DataFrame(results_feat)

    return results_clust, results_feat


if __name__ == '__main__':

    from params import ITERS
    from params import params

    results_clust, results_feat = [], []
    for params in tqdm(params):

        method, n, k, target_col, bonf_correct = params

        results_clust_, results_feat_ = simulate_experiment(
            # Run experiment with `method`
            method, n, k, target_col, bonf_correct=bonf_correct, iters=ITERS
        )

        results_clust.append(results_clust_)
        results_feat.append(results_feat_)

    results_clust = pd.concat(results_clust, axis=0)
    results_feat = pd.concat(results_feat, axis=0)
    f_out = Path(__file__).parent / 'results_clust.csv'
    results_clust.to_csv(f_out, index=False)
    f_out = Path(__file__).parent / 'results_feat.csv'
    results_feat.to_csv(f_out, index=False)
