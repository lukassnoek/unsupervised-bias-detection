from itertools import product

ITERS = 500

N = [50, 500, 1000]
K = [1, 5, 50, 100]
target_col = ['y_pred', 'fp', 'fn', 'err']#, 'fp_prec', 'fn_rec']
bonf_correct = [True, False]
method = ['hbac', 'randomclusters']
params = list(product(method, N, K, target_col, bonf_correct))
