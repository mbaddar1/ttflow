import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.datasets import make_circles
from tqdm import tqdm
from loguru import logger

GB = 1024.0 ** 3

import sys
import numpy as np
import torch

def get_inmemory_size_mb(obj, seen=None):
    """
    Recursively calculates the true in-memory size (MB) of an unpickled object,
    correctly reading buffer memory for NumPy arrays and PyTorch tensors.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0.0
    seen.add(obj_id)

    # PyTorch Tensors (CUDA or CPU)
    if isinstance(obj, torch.Tensor):
        return (obj.nelement() * obj.element_size()) / (1024 ** 2)

    # NumPy ndarray
    if isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 ** 2)

    # Dictionaries (keys + values)
    if isinstance(obj, dict):
        return (
            sys.getsizeof(obj) / (1024 ** 2)
            + sum(get_inmemory_size_mb(k, seen) + get_inmemory_size_mb(v, seen) for k, v in obj.items())
        )

    # Iterables (lists, tuples, sets)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return (
            sys.getsizeof(obj) / (1024 ** 2)
            + sum(get_inmemory_size_mb(item, seen) for item in obj)
        )

    # Fallback for standard Python primitives
    return sys.getsizeof(obj) / (1024 ** 2)

def check_system_memory_and_swap(min_ram_gb, min_swap_gb):
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    logger.info(f"ram = {np.round(ram.total/GB,2)}, swap = {np.round(swap.total/GB,2)} GB")
    ram_ok = ram.total >= min_ram_gb * GB
    swap_ok = swap.total >= min_swap_gb * GB
    return ram_ok and swap_ok


# def check_tensor_validity(X: torch.Tensor):
#     return not (torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)))
#
#
# def check_and_process(X0: torch.Tensor, X1: torch.Tensor):
#     if check_tensor_validity(X1):
#         return X0, X1
#     else:
#         X0, X1 = impute_samples(X0, X1)
#         return X0, X1
#
#
# def impute_samples(X0: torch.Tensor, X1: torch.Tensor):
#     """
#     Why : to avoid nan issues
#     https://discuss.pytorch.org/t/fail-to-run-torch-linalg-svd-because-error-11/150418
#     @param X0: Assumed to have no nan/inf
#     @param X1: The one to impute nan/inf
#     @return:
#     """
#     # Impute X1 and resample X0 accordingly
#     X1_imputed = impute(tns=X1)
#     perm = torch.randperm(X1_imputed.size(0))
#     idx = perm[:(X1_imputed.shape[0])]
#     X0_resampled = X0[idx]
#     n_dims_X0 = len(X0_resampled.shape)
#     n_dims_X1 = len(X1_imputed.shape)
#     assert n_dims_X0 == n_dims_X1
#     for i in range(n_dims_X1):
#         assert X0_resampled.shape[i] == X1_imputed.shape[i]
#     return X0_resampled, X1_imputed


def clean(X: pd.DataFrame) -> pd.DataFrame:
    # D = tns.shape[1]
    # new_tns_list = []
    # for d in range(D):
    #     col_mean = torch.nanmean(input=tns[:, d], dim=0).item()
    #     new_tns_list.append(torch.nan_to_num(input=tns[:, d], nan=col_mean))
    # new_tns = torch.stack(tensors=new_tns_list, dim=1)
    # assert not torch.any(torch.isnan(new_tns))
    # assert not torch.any(torch.isinf(new_tns))
    # return new_tns
    N = X.shape[0]
    index_list = []
    logger.info(f"Imputing Dataframe with shape {X.shape}")
    for i in tqdm(range(N), desc="Remove bad rows"):
        cond = not (np.any(np.isnan(X.iloc[i, :])) or np.any(np.isinf(X.iloc[i, :]))).item()
        index_list.append(cond)
    new_X = X.iloc[index_list, :]
    logger.info(f"After imputation # rows reduced from {X.shape[0]} to {new_X.shape[0]} "
                f": {(X.shape[0] - new_X.shape[0])} rows were removed")
    return new_X


def remove_outliers_quantile(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Outlier removal based on 95% for each dimension

    x : torch.Tensor of the shape N X D
    """
    D = x.shape[1]
    N = x.shape[0]
    q_ = torch.quantile(input=x, q=torch.tensor([alpha, 1 - alpha], dtype=x.dtype), dim=1)
    filter_idx = torch.tensor(data=[True] * N)
    for d in range(D):
        q_min = q_[0, d].item()
        q_max = q_[1, d].item()
        idx1 = torch.greater_equal(input=x[:, d], other=q_min)
        idx2 = torch.less_equal(input=x[:, d], other=q_max)
        idx = torch.bitwise_and(idx1, idx2)
        filter_idx = torch.bitwise_and(idx, filter_idx)

    x_filter_idx = x[filter_idx, :]
    return x_filter_idx


def remove_outliers_range(x: torch.Tensor, x_min, x_max):
    """
        Outlier removal based on a given explicit range

        x : torch.Tensor of the shape N X D
        """
    D = x.shape[1]
    N = x.shape[0]
    filter_idx = torch.tensor(data=[True] * N)
    for d in range(D):
        idx1 = torch.greater_equal(input=x[:, d], other=x_min[d].item())
        idx2 = torch.less_equal(input=x[:, d], other=x_max[d].item())
        idx = torch.bitwise_and(idx1, idx2)
        filter_idx = torch.bitwise_and(idx, filter_idx)

    x_filter_idx = x[filter_idx, :]
    return x_filter_idx


if __name__ == '__main__':
    n_samples = 10000
    x_train = torch.tensor(make_circles(n_samples=n_samples, shuffle=True, noise=0.05, factor=0.3)[0])
    x_test = torch.tensor(make_circles(n_samples=n_samples, shuffle=True, noise=0.05, factor=0.3)[0])
    x_min, x_max = torch.min(input=x_train, dim=0).values, torch.max(input=x_train, dim=0).values
    # apply outlier removal
    x1 = remove_outliers_quantile(x_test, alpha=0.001)
    x2 = remove_outliers_range(x=x_test, x_min=x_min, x_max=x_max)
