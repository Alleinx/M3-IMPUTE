import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.linear_regression import linear_regression
from uci.uci_data import load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='uci')
    parser.add_argument('--data', type=str, default='housing')
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--split_sample', type=float, default=0.)
    parser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    parser.add_argument('--split_train', action='store_true', default=False)
    parser.add_argument('--split_test', action='store_true', default=False)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot

    parser.add_argument('--method', type=str, default='m3-impute')    # default is m3-impute
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--best_level', action='store_true', default=True)
    parser.add_argument('--comment', type=str, default='v1')
    
    parser.add_argument('--corrupt', type=str, default="mcar")
    parser.add_argument('--masking_distribution', type=str, default='uniform') # 1 - edge dropout rate

    args = parser.parse_args() 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for args.data in ['yacht', 'wine', 'concrete', 'energy', 'housing', 'kin8nm', 'naval', 'power']:
        print('=' * 100)
        print(f'[Config] dataset:{args.data}')

        for i in range(5):
            args.seed = i

            # Loading data and create missing mask
            data = load_data(args)

            if not args.method.startswith('gnn'):
                best_levels = {'mean':0, 'knn':3, 'svd':2, 'mice':2, 'spectral':1}       # The i-th HPO setting of the corresponding method has the best performance
                args.level = best_levels[args.method] if args.method in best_levels else None

            # create saving results dir:
            log_path = './uci/y_results/results/{}_{}/{}/{}/'.format(args.method, args.comment, args.data, args.seed)
            if not os.path.isdir(log_path):
                os.makedirs(log_path)

            # loading pretrained model param from 
            # NOTE: should modify this path to where you store your model parameters
            load_path = './uci/test/{}/EXP1_impute/'.format(args.data)

            # First perform imputation then do Linear Regression on completed matrix
            linear_regression(data, args, log_path, load_path)

if __name__ == '__main__':
    main()

