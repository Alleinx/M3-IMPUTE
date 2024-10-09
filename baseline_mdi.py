import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from uci.uci_subparser import add_uci_subparser
from training.baseline import baseline_mdi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')


    parser.add_argument('--masking_distribution', type=str, default='uniform')
    parser.add_argument('--corrupt', type=str, default="mcar")
    parser.add_argument('--mar_rate_obs', type=float, default=0.1)
    parser.add_argument('--mar_rate_missing', type=float, default=0.15)
    parser.add_argument('--mnar_known_mask', type=float, default=0.1)


    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args)

    log_path = './{}/test/{}/{}_{}/'.format(args.domain,args.data,args.method,args.log_dir)

    res = []
    for i in range(5):
        seed = i
        np.random.seed(seed)
        torch.manual_seed(seed)
        mae = baseline_mdi(data, args, log_path)
        res.append(mae)
    
    print(f'Mean MAE (x 10):{np.mean(res)}, std: {np.std(res)}')


if __name__ == '__main__':
    main()