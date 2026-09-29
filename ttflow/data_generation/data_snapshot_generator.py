"""
This script is to generate snapshots of data-snapshots to train TT-RecFlow both based on the space-time (param-sharing)
architecture and the time-grid based one
The main goal to have a fixed dataset-snapshot for each distribution/dataset-family for experiments reproducibility

To avoid large files in GitHub Repo, the data-snapshot files are saved here
https://drive.google.com/drive/folders/1wazn94CNAhzXrKliK0UjWIiPMzcw6O_U?usp=drive_link

sample runs
python data_snapshot_generator.py --dataset-name skewed_gaussian_2d --mean [0.0,0.0] --cov [[1.0,0.0],[0.0,1.0]] --shape [5.0,5.0]
python data_snapshot_generator.py --dataset-name gaussian_mixture_2d --gm-ncomp 16

"""
import json
import os.path
import pickle
import argparse
import numpy as np
from loguru import logger
from datetime import datetime

from ttflow.data_generation.datasnapshot_utils import DataSnapShotUtils
from ttflow.utils import check_system_memory_and_swap

SEED = 42
MIN_MEM_GB = 30
MIN_SWAP_GB = 32
TIMESTAMP = datetime.now().isoformat()
DATASETS_SNAPSHOTS_DIR = "data-snapshots/experimental"

parser = argparse.ArgumentParser()
dataset_name_choices = ["manywell", "gaussian_mixture_2d", "skewed_gaussian_2d"]
# main args
parser.add_argument("--n-train-test", type=int, default=500_000,
                    help="Number of training samples (keep large, may subsample later)", required=False)
parser.add_argument("--n-eval", type=int, default=200_000,
                    help="Number of test samples", required=False)
parser.add_argument("--n-time-grid", type=int, default=1024,
                    help="Number of points in the time grid", required=False)
parser.add_argument("--dataset-name", choices=dataset_name_choices, type=str, default="manywell",
                    help="Name of the dataset", required=False)
# dataset specific
parser.add_argument("--manywell-total-data-dim", type=int, default=4,
                    help="Dimensionality of the data for manywell dist", required=False)
parser.add_argument("--manywell-dim", type=int, default=4,
                    help="Dimensionality of the ManyWell target", required=False)
# for gaussian mixture
parser.add_argument("--gm-ncomp", type=int, default=3, required=False)
# for multivariate skewed gaussian (2D)
# sample run
# data_snapshot_generator.py --dataset-name skewed_gaussian_2d --mean [0.0,0.0] --cov [[1.0,0.0],[0.0,1.0]] --shape [5.0,5.0]
parser.add_argument(
    "--mean",
    type=json.loads,
    default=[0.0, 0.0],
    help="2D mean/center vector [mu_x, mu_y] (e.g., --mean 1.0 -0.5)",
)

parser.add_argument(
    "--cov",
    type=json.loads,
    default=[[1.0, 0.0], [0.0, 1.0]],
    help="Covariance matrix as a JSON list of lists, e.g. '[[1, 0.5], [0.5, 1]]'",
)
parser.add_argument(
    "--shape",
    type=json.loads,
    default=[0.0, 0.0],
    help="2D skewness shape vector alpha (e.g., --shape 2.0 4.5)",
)

args = parser.parse_args()

dataset_name = args.dataset_name
if __name__ == "__main__":
    logger.info("Parsing arguments...")
    args = parser.parse_args()
    n_train_test = args.n_train_test
    n_eval = args.n_eval
    n_time_grid = args.n_time_grid
    time_grid = np.linspace(0, 1, n_time_grid)
    dataset_name = args.dataset_name
    logger.info("Memory size check")
    # checking memory
    assert check_system_memory_and_swap(min_ram_gb=MIN_MEM_GB, min_swap_gb=MIN_SWAP_GB), "low memory, increase! "
    logger.info(f"Generating data-snapshot with dataset_name = {dataset_name} n_train_test = {n_train_test}, "
                f"n_eval = {n_eval}, n_time_grid = {n_time_grid}, time_grid.shape = {time_grid.shape}")
    if dataset_name == "manywell":
        assert (hasattr(args, "manywell_total_data_dim") and hasattr(args,
                                                                     "manywell_dim") and
                args.manywell_total_data_dim is not None and args.manywell_dim is not None)

        data_snapshot_dict = (DataSnapShotUtils.
                              generate_manywell_snapshot(total_data_dim=args.manywell_total_data_dim,
                                                         manywell_dim=args.manywell_dim,
                                                         n_train_test=args.n_train_test,
                                                         n_eval=args.n_eval,
                                                         time_grid=time_grid))

        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_data_dim_{args.manywell_total_data_dim}"
                              f"_manywell_dim_{args.manywell_dim}"
                              f"_n_train_test_{args.n_train_test}_n_eval_{args.n_eval}"
                              f"_ntimegrid_{n_time_grid}_timestamp_{TIMESTAMP}.pkl")

    # TODO add GM
    elif dataset_name == "gaussian_mixture_2d":
        data_dim = 2
        n_comp = args.gm_ncomp if hasattr(args, "gm_ncomp") and args.gm_ncomp is not None else 3
        data_snapshot_dict = DataSnapShotUtils.generate_gaussian_mixture_2d(n_train_test=args.n_train_test,
                                                                            num_components=n_comp,
                                                                            n_eval=args.n_eval,
                                                                            time_grid=time_grid)
        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_ncomp_{n_comp}"
                              f"_data_dim_{data_dim}"
                              f"_n_comp_{n_comp}"
                              f"_n_train_test_{n_train_test}_n_eval_{args.n_eval}"
                              f"_ntimegrid_{n_time_grid}_timestamp_{TIMESTAMP}.pkl")
    # TODO add skewed Gaussian
    elif dataset_name == "skewed_gaussian_2d":
        dist_params = ["mean", "cov", "shape"]
        assert all(dist_param in dist_params for dist_param in ["mean", "cov", "shape"]), \
            f"for dataset_name = {dataset_name}, must pass params {dist_params}"
        x_dim = 2
        y_dim = 2
        kwargs = {"mean": args.mean, "cov": args.cov, "shape": args.shape}
        data_snapshot_dict = DataSnapShotUtils.generate_data_snapshot(dataset_name=dataset_name,
                                                 x_dim=x_dim, y_dim=y_dim,
                                                 n_train_test=args.n_train_test,
                                                 n_eval=n_eval, time_grid=time_grid, **kwargs)
        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_mean_{str(kwargs["mean"])}"
                              f"_cov_{str(kwargs["cov"])}"
                              f"_shape_{kwargs["shape"]}"
                              f"_n_train_test_{n_train_test}_n_eval_{args.n_eval}"
                              f"_ntimegrid_{n_time_grid}_timestamp_{TIMESTAMP}.pkl")
    else:
        raise ValueError(f"dataset_name {dataset_name} not supported")
    logger.info(f"Saving snapshot file...")
    snapshot_filepath = os.path.join(DATASETS_SNAPSHOTS_DIR, snapshot_file_name)
    with open(snapshot_filepath, "wb") as f:
        pickle.dump(data_snapshot_dict, f)
        f.flush()
        f.close()
    logger.info(f"Successfully saved snapshot file to path = {snapshot_filepath}")
