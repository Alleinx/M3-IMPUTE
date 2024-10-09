from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD, SoftImpute
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
import time

from utils.utils import construct_missing_X_from_mask
import os

def baseline_mdi(data, args, log_path):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    t0 = time.time()
    train_edge_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
    if hasattr(args,'split_sample') and args.split_sample > 0.:
        if args.split_test:
            higher_y_index = data.higher_y_index
            X = X[higher_y_index]
            X_incomplete = X_incomplete[higher_y_index]
    t_load = time.time()

    X_filled = baseline_inpute(X_incomplete, args.method,args.level)
    t_impute = time.time()

    if hasattr(args,'split_sample') and args.split_sample > 0.:
        if not args.split_test:
            higher_y_index = data.higher_y_index
            X = X[higher_y_index]
            X_incomplete = X_incomplete[higher_y_index]
            X_filled = X_filled[higher_y_index]

    mask = np.isnan(X_incomplete)
    diff = X[mask] - X_filled[mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    t_test = time.time()

    # obj = dict()
    # obj['args'] = args
    # obj['rmse'] = rmse
    # obj['mae'] = mae
    # obj['load_time'] = t_load - t0
    # obj['impute_time'] = t_impute - t_load
    # obj['test_time'] = t_test - t_impute
    # print('rmse: {:.3g}, mae: {:.3g}'.format(rmse,mae))
    # pickle.dump(obj, open(log_path + 'result.pkl', "wb"))
    return mae * 10

def baseline_inpute(X_incomplete, method='mean',level=0):

    if method == 'mean':
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        return X_filled_mean
    elif method == 'knn':
        k = [3,5,10,20][level]
        X_filled_knn = KNN(k=k, verbose=False).fit_transform(X_incomplete)
        return X_filled_knn
    elif method == 'svd':
        rank = [np.ceil((X_incomplete.shape[1]-1)/10),np.ceil((X_incomplete.shape[1]-1)/5),X_incomplete.shape[1]-1][level]
        X_filled_svd = IterativeSVD(rank=int(rank),verbose=False).fit_transform(X_incomplete)
        return X_filled_svd
    elif method == 'mice':
        max_iter = [3,10,50][level]
        X_filled_mice = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete)
        return X_filled_mice
    elif method == 'spectral':
        # default value for the sparsity level is with respect to the maximum singular value,
        # this is now done in a heuristic way
        sparsity = [0.5,None,3,][level]
        X_filled_spectral = SoftImpute(shrinkage_value=sparsity).fit_transform(X_incomplete)
        return X_filled_spectral
    else:
        from hyperimpute.plugins.imputers import Imputers
        input_missing_x = pd.DataFrame(X_incomplete)
        out = []
        if method == 'miracle':
            params = {
                "window": 10,       # best of [10, 20],
                "n_hidden": 8,      # best of [8,  16, 32, 64], 
                "max_steps": 2000, # best of [500, 1000, 2000]
                # "reg_lambda": [0.2, 0.5, 0.7], no diff
                # "reg_beta": [0.2, 0.5, 0.7], no diff
                # "reg_m": [0.2, 0.5, 0.7], no diff
                # "DAG_only": True,
            }
        elif method == 'miwae':
            params = {
                "n_epochs": 3000,         # best of [1000, 2000, 3000]
                # "latent_size": [1, 5, 10],
                # "n_hidden": int = 1,
                "K": 10,                 # [5, 10, 15, 20],
                # random_state: int = 0,
                # batch_size: int = 256,
            }
        elif method == 'gain':
            params = {
                # batch_size: 256, # no diff
                "n_epochs": 3000,       # best of [1000, 2000, 3000], # no diff
                # "hint_rate": [0.8, 0.9, 1], # no diff
                # "loss_alpha": [5, 10, 20, 30], # no diff
            }

        # combinations = list(itertools.product(*params.values()))
        # result = [dict(zip(params.keys(), combo)) for combo in combinations] 
        
        # for config in result:
        plugin = Imputers().get(
            method,
            **params
        )
        
        impute_res = plugin.fit_transform(input_missing_x.copy()).to_numpy()

        return impute_res