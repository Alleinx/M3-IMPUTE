import torch.optim as optim
import numpy as np
import os.path as osp
import torch
import subprocess
import scipy.stats as stats
from scipy import optimize

def np_random(seed=None):
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def save_mask(length,true_rate,log_dir,seed):
    np.random.seed(seed)
    mask = np.random.rand(length) < true_rate
    np.save(osp.join(log_dir,'len'+str(length)+'rate'+str(true_rate)+'seed'+str(seed)),mask)
    return mask

def _mar_logistic_masking(X, rate_obs, rate_missing):
    def pick_coefficients(X, idxs_obs=None, idxs_nas=None, self_mask=False):
        n, d = X.shape
        if self_mask:
            coeffs = torch.randn(d)
            Wx = X * coeffs
            coeffs /= torch.std(Wx, 0)
        else:
            d_obs = len(idxs_obs)
            d_na = len(idxs_nas)
            coeffs = torch.randn(d_obs, d_na)
            Wx = X[:, idxs_obs].mm(coeffs)
            coeffs /= torch.std(Wx, 0, keepdim=True)
        return coeffs

    def fit_intercepts(X, coeffs, p, self_mask=False):
        if self_mask:
            d = len(coeffs)
            intercepts = torch.zeros(d)
            for j in range(d):

                def f(x):
                    return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

                intercepts[j] = optimize.bisect(f, -50, 50)
        else:
            d_obs, d_na = coeffs.shape
            intercepts = torch.zeros(d_na)
            for j in range(d_na):

                def f(x):
                    return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

                intercepts[j] = optimize.bisect(f, -50, 50)
        return intercepts

    X = X.to_numpy()
    assert len(X.shape) == 2, "X should be 2 dimensional"
    n, d = X.shape

    ori_type_is_np = isinstance(X, np.ndarray)
    if ori_type_is_np:
        X = torch.from_numpy(X).to(torch.float32)
    else:
        X = torch.clone(X).to(torch.float32)

    assert (
        torch.isnan(X).sum() == 0
    ), "the input X of the mar_logistic() shouldn't containing originally missing data"

    mask = torch.ones(n, d).bool()

    # number of variables that will have no missing values (at least one variable)
    d_obs = max(int(rate_obs * d), 1)
    d_na = d - d_obs  # number of variables that will have missing values

    # Sample variables will all be observed, and the left will be with missing values
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coefficients(X, idxs_obs, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, rate_missing)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)
    # print(ps)
    ber = torch.rand(n, d_na)
    # print(ber)
    # print(ber >= ps)
    mask[:, idxs_nas] = ber >= ps  # True mean mask
    print(mask)
    return mask

def train_test_mask(known_prob, edge_num, mode="mcar", mask_dist="uniform", X=None, args=None):
    if mode == "mcar":
        known_mask = get_known_mask(known_prob, edge_num, mask_dist)
    elif mode == "mar" and X is not None:
        # rate_obs = 0.1
        # rate_missing = 0.15
        rate_obs     = args.mar_rate_obs
        rate_missing = args.mar_rate_missing
        known_mask = _mar_logistic_masking(X, rate_obs, rate_missing)
        # print(f"MAR mask shape {known_mask.shape}")
        known_mask = known_mask.view(-1)
        # print(f"MAR mask shape (after flatten) {known_mask.shape}")
        print(f"MAR known ratio: {known_mask.float().mean().item()}")
    elif mode == "mnar":
        rate_obs     = args.mar_rate_obs
        rate_missing = args.mar_rate_missing
        known_mask = _mar_logistic_masking(X, rate_obs, rate_missing)
        # added for mnar
        known_mask = known_mask.view(-1)
        mnar_addintional_known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < args.mnar_known_mask).view(-1)
        known_mask = known_mask * mnar_addintional_known_mask
        print(f"MNAR known ratio: {known_mask.float().mean().item()}")
    else:
        raise AttributeError(f"'{mode}' is not a valid masking mode")
    # print(f"Mode: {mode} - Known: {known_prob} - Real Masking Ratio: {1 - known_mask.float().mean().item()}")
    return known_mask

def get_known_mask(known_prob, edge_num, mode="uniform"):
    if mode == "uniform":
        known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    elif mode == "normal":
        normal_dist = torch.randn(edge_num, 1)
        threshold = stats.norm.ppf(1 - known_prob)
        known_mask = (normal_dist > threshold).view(-1)
    else:
        raise AttributeError(f"'{mode}' is not a valid masking mode")
    # print(f"Mode: {mode} - Known: {known_prob} - Real Masking Ratio: {1 - known_mask.float().mean().item()}")
    # print('during training:', known_mask.float().mean().item())
    return known_mask
    

def mask_edge(edge_index,edge_attr,mask,remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr

def one_hot(batch,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,torch.tensor(batch,dtype=int))

def soft_one_hot(batch,depth):
    batch = torch.tensor(batch)
    encodings = torch.zeros((batch.shape[0],depth))
    for i,x in enumerate(batch):
        for r in range(depth):
            encodings[i,r] = torch.exp(-((x-float(r))/float(depth))**2)
        encodings[i,:] = encodings[i,:]/torch.sum(encodings[i,:])
    return encodings

def construct_missing_X_from_mask(train_mask, df):
    nrow, ncol = df.shape
    data_incomplete = np.zeros((nrow, ncol))
    data_complete = np.zeros((nrow, ncol)) 
    train_mask = train_mask.reshape(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            data_complete[i,j] = df.iloc[i,j]
            if train_mask[i,j]:
                data_incomplete[i,j] = df.iloc[i,j]
            else:
                data_incomplete[i,j] = np.NaN
    return data_complete, data_incomplete

def construct_missing_X_from_edge_index(train_edge_index, df):
    nrow, ncol = df.shape
    data_incomplete = np.zeros((nrow, ncol))
    data_complete = np.zeros((nrow, ncol)) 
    train_edge_list = torch.transpose(train_edge_index,1,0).numpy()
    train_edge_list = list(map(tuple,[*train_edge_list]))
    for i in range(nrow):
        for j in range(ncol):
            data_complete[i,j] = df.iloc[i,j]
            if (i,j) in train_edge_list:
                data_incomplete[i,j] = df.iloc[i,j]
            else:
                data_incomplete[i,j] = np.NaN
    return data_complete, data_incomplete

# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda