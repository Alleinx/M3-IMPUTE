from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import pickle
import joblib
import time
import os.path
from os import path

from training.baseline import baseline_impute
from utils.utils import construct_missing_X_from_mask

from training.gnn_mdi import get_known_A_during_training, eliminate_feature_index, reconstruct_known_missing_A

def linear_regression(data, args, log_path, load_path):
    t0 = time.time()
    n_row, n_col = data.df_X.shape
    x = data.x.clone().detach()
    edge_index = data.edge_index.clone().detach()
    train_edge_mask = data.train_edge_mask.numpy()
    train_edge_index = data.train_edge_index.clone().detach()
    train_edge_attr = data.train_edge_attr.clone().detach()
    test_edge_index = data.test_edge_index.clone().detach()
    test_edge_attr = data.test_edge_attr.clone().detach()

    y = data.y.detach().numpy()
    train_y_mask = data.train_y_mask.clone().detach()
    test_y_mask = data.test_y_mask.clone().detach()
    y_train = y[train_y_mask]
    y_test = y[test_y_mask]


    if args.method == 'm3-impute':
        device = torch.device('cuda:1')

        model = torch.load(load_path+'model.pt',map_location=device)
        model.eval()

        impute_model = torch.load(load_path+'impute_model.pt',map_location=device)
        impute_model.eval()

        # Prepare required attr:
        # =======================
        num_nodes, num_feature = data.x.shape
        num_record = num_nodes - num_feature
        # print(num_record, num_feature)

        # 1. obtain the idx for the last record
        tgt_v = torch.tensor([num_record], dtype=int, device=torch.device('cpu'))
        idx = eliminate_feature_index(train_edge_index[0], tgt_v).to(device)
        # 2. reconstruct adjacency matrix (Num_sample, num_feature)
        # (1) Observable: fill in with edge value
        # (2) Missing:    fill in with epsilon
        A, Known_Mask = reconstruct_known_missing_A((num_record, num_feature), train_edge_index.to(device), train_edge_attr.to(device), num_record, device=device, idx=idx, epsilon=1e-4)
        # =======================

        t_load = time.time()

        # Start Impute:
        # =======================
        with torch.no_grad():
            # compute Graph rep 
            x_embd = model(x.to(device), train_edge_attr.to(device), train_edge_index.to(device), A.to(device)).to(device)
            # ----------------------------------
            # Impute
            test_e = test_edge_index.to(device)
            KM = Known_Mask.to(device)
            x_pred, _ = impute_model(obs_nodes_embs=x_embd[: num_record].to(device), fea_nodes_embs=x_embd[num_record:].to(device),known_edges=train_edge_index.to(device), impute_target_edges=test_e, known_mask=KM)
            t_impute = time.time()

            x_pred = x_pred[:int(test_edge_attr.shape[0] / 2)]
            X_true, X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
            X = X_incomplete
            # replace np.nan with imputed values:
            for i in range(int(test_edge_attr.shape[0] / 2)):
                assert X_true[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] == test_edge_attr[i]
                X[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] = x_pred[i]
    else:
        X_true, X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
        t_load = time.time()

        X = baseline_impute(X_incomplete, args.method, args.level)
        t_impute = time.time()
    # ==========================
    # LR prediction w/ completed data matrix:
    reg = LinearRegression().fit(X[train_y_mask, :], y_train)
    y_pred_test = reg.predict(X[test_y_mask, :])
    t_reg = time.time()

    rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
    mae = np.mean(np.abs(y_pred_test - y_test))
    t_test = time.time()

    if path.exists(log_path + 'result.pkl'):
        obj = joblib.load(log_path + 'result.pkl')
        obj['args_linear_regression'] = args
    else:
        obj = dict()
        obj['args'] = args

    obj['load_path'] = load_path
    obj['rmse'] = rmse
    obj['mae'] = mae
    obj['load_time'] = t_load - t0
    obj['impute_time'] = t_impute - t_load
    obj['reg_time'] = t_reg - t_impute
    obj['test_time'] = t_test - t_reg
    print('{}: rmse: {:.3g}, mae: {:.3g}'.format(args.method,rmse,mae))
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))
