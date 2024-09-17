from copy import copy
from itertools import product
from collections import defaultdict
import warnings
warnings.filterwarnings("error")

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from unsupervised_bias_detection.clustering import BiasAwareHierarchicalKMeans


def simulate_null_data(N, K, target_col, binary_target=True):

    if binary_target:
        y = np.random.choice([0, 1], size=N)
        model = make_pipeline(StandardScaler(), LogisticRegression())
    else:
        y = np.random.randn(N)
        model = make_pipeline(StandardScaler(), LinearRegression())

    X = np.random.randn(N, K)
    cols = [f'feat_{i}' for i in range(K)]
    y_pred = cross_val_predict(model, X, y)

    if binary_target:
        err = (y != y_pred).astype(int)
        fp = np.logical_and(y == 0, y_pred == 1).astype(int)
        fn = np.logical_and(y == 1, y_pred == 0).astype(int)
        X = pd.DataFrame(
            np.c_[X, y, y_pred, err, fp, fn],
            columns=cols + ['y', 'y_pred', 'err', 'fp', 'fn']
        )
    else:
        err = y - y_pred
        X = pd.DataFrame(
            np.c_[X, y, y_pred, err],
            columns=cols + ['y', 'y_pred', 'err']
        )
    
    target = X[target_col]
    target.name = 'performance metric'  # to make sure webapp works

    if binary_target:
        X = X.drop(['y', 'y_pred', 'err', 'fp', 'fn'], axis=1)
    else:
        X = X.drop(['y', 'y_pred', 'err'], axis=1)
 
    return X, target


def check_before_test(c0, c1, min_samples=3):

    if (c0.size < min_samples) or (c1.size < min_samples):
        return True

    if np.mean(c0) == np.mean(c1):
        return True
    
    if np.all(c0 == c0[0]) or np.all(c1 == c1[0]):
        return True

    return False


def simulate_experiment(N, K, target_col, binary_target, min_cluster_size, bonf_correct=False, alpha=0.05, iters=100, verbose=False):
    
    sig_clust = np.zeros(iters)
    sig_feat = np.zeros(iters)
    hbac = BiasAwareHierarchicalKMeans(n_iter=5, min_cluster_size=min_cluster_size)

    for i in range(iters):
        X, target = simulate_null_data(N, K, target_col, binary_target)
        X = X.to_numpy()
        target = target.to_numpy()
        hbac.fit(X, target)

        clust_p = []
        diff, label = 0, 0
        for ii, l in enumerate(np.unique(hbac.labels_)):
            c0 = target[hbac.labels_ != l]
            c1 = target[hbac.labels_ == l]

            should_continue = check_before_test(c0, c1)
            if should_continue:
                continue

            _, p = ttest_ind(c0, c1, equal_var=False)
            clust_p.append(p)
            this_diff = c0.mean() - c1.mean()

            if np.abs(this_diff) > diff:
                diff = copy(this_diff)
                label = copy(l)
        
        idx = hbac.labels_ == label#np.random.choice(hbac.labels_)
        feat_p = []
        for ii in range(K):
            X_ = X[:, ii]
            c0 = X_[idx]
            c1 = X_[~idx]

            should_continue = check_before_test(c0, c1)
            if should_continue:
                continue
            
            _, p = ttest_ind(c0, c1)
            if np.isnan(p):
                print(c0, c1)
            feat_p.append(p)

        feat_p = np.array(feat_p)
        alpha_f = alpha / K if bonf_correct else alpha
        sig_feat[i] = (feat_p < alpha_f).sum() / K
        
        clust_p = np.array(clust_p)
        alpha_f = alpha / hbac.n_clusters_ if bonf_correct else alpha
        sig_clust[i] = (clust_p < alpha_f).sum() / hbac.n_clusters_

    fpr_feat = (sig_feat > 0).mean()
    ev_feat = sig_feat.mean()

    fpr_clust = (sig_clust > 0).mean()
    ev_clust = sig_clust.mean() 

    if verbose:
        print(f"FPR (cluster): {fpr_clust:.3f}")
        print(f"Exp. value (cluster): {ev_clust:.3f}")
        print(f"FPR (feature): {fpr_feat:.3f}")
        print(f"Exp. value (feature): {ev_feat:.3f}")

    return fpr_feat, ev_feat, fpr_clust, ev_clust 


if __name__ == '__main__':

    ITERS = 100
    ALPHA = 0.05
    BONF_CORRECT = False

    N = [50, 100, 500, 1000]
    K = [1, 5, 10, 50, 100]
    target_col = ['y', 'y_pred', 'fp', 'fn', 'err']
    binary_target = [True]
    min_cluster_size = [5, 10, 50, 100]

    params = list(product(N, K, target_col, binary_target, min_cluster_size))    
    results = defaultdict(list)
    for params in tqdm(params):
        n, k, target_col_, binary_target_, min_cluster_size_ = params

        if not binary_target_ and target_col_ in ['fp', 'fn']:
            continue

        if n <= min_cluster_size_:
            continue

        fpr_feat, ev_feat, fpr_clust, ev_clust = simulate_experiment(
            n, k, target_col_, binary_target_, min_cluster_size_,
            bonf_correct=BONF_CORRECT, alpha=ALPHA, iters=ITERS
        )
        
        results['fpr_feat'].append(fpr_feat)
        results['ev_feat'].append(ev_feat)
        results['fpr_clust'].append(fpr_clust)
        results['ev_clust'].append(ev_clust)

        for p_name, p in zip(['N', 'K', 'target', 'binary', 'min_cluster_size'], params):
            results[p_name].append(p)

    results = pd.DataFrame(results)
    results.to_csv('./results_.csv', index=False)