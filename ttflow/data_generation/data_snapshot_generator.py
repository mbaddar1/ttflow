"""
This script is to generate snapshots of data-snapshots to train TT-RecFlow both based on the space-time (param-sharing)
architecture and the time-grid based one
The main goal to have a fixed dataset-snapshot for each distribution/dataset-family for experiments reproducibility

To avoid large files in GitHub Repo, the data-snapshot files are saved here
https://drive.google.com/drive/folders/1wazn94CNAhzXrKliK0UjWIiPMzcw6O_U?usp=drive_link

sample runs
python data_snapshot_generator.py --dataset-name skewed_gaussian_2d --mean [0.0,0.0] --cov [[1.0,0.0],[0.0,1.0]] --shape [5.0,5.0]
python data_snapshot_generator.py --dataset-name gaussian_mixture_2d --gaussian-mixture-2d-num-components 4 --n-train 100_000 --n-test 20_000 --n-eval 20_000

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
from ttflow.viz.viz import Visualizer

SEED = 42
MIN_MEM_GB = 30
MIN_SWAP_GB = 32
TIMESTAMP = datetime.now().isoformat()
DATASETS_SNAPSHOTS_DIR = "data-snapshots/experimental"
DATASET_FIGURES_PATH = "figures/data_generation"
parser = argparse.ArgumentParser()
dataset_name_choices = ["manywell", "gaussian_mixture_2d", "skewed_gaussian_2d","circles"]
# args names

# main args
parser.add_argument("--n-train", type=int, default=500_000,
                    help="Number of training samples (keep large, may subsample later)", required=False)
parser.add_argument("--n-test", type=int, default=200_000,
                    help="Number of training samples (keep large, may subsample later)", required=False)
parser.add_argument("--n-eval", type=int, default=200_000,
                    help="Number of test samples", required=False)
# TODO to remove
# parser.add_argument("--n-time-grid", type=int, default=1024,
#                     help="Number of points in the time grid", required=False)
parser.add_argument("--dataset-name", choices=dataset_name_choices, type=str, default="manywell",
                    help="Name of the dataset", required=False)
# dataset specific params
# manywell
parser.add_argument("--manywell-data-dim", type=int, default=4,
                    help="Dimensionality of the data for manywell dist", required=False)
parser.add_argument("--manywell-dim", type=int, default=4,
                    help="Dimensionality of the ManyWell target", required=False)
# for gaussian mixture
parser.add_argument("--gm-2d-num-components", type=int, default=3,
                    required=False)
parser.add_argument("--gm-2d-radius", type=float, default=10.0,
                    required=False)
parser.add_argument("--gm-2d-variance", type=float, default=0.3,
                    required=False)

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
    dataset_name = args.dataset_name
    logger.info("Memory size check")
    # checking memory
    assert check_system_memory_and_swap(min_ram_gb=MIN_MEM_GB, min_swap_gb=MIN_SWAP_GB), "low memory, increase! "
    logger.info(
        f"Generating data-snapshot with dataset_name = {dataset_name} "
        f"with n_train/test/eval = {args.n_train}, {args.n_test}, {args.n_eval}")

    if dataset_name == "manywell":
        assert (hasattr(args, "manywell_total_data_dim") and hasattr(args,
                                                                     "manywell_dim") and
                args.manywell_total_data_dim is not None and args.manywell_dim is not None)
        # FIXME : for many-well, the dim info is taken from the kwargs not the x_dim and y_dim
        data_snapshot_dict = DataSnapShotUtils.generate_transport_data_snapshot(dataset_name=dataset_name, x_dim=-1,
                                                                                y_dim=-1, n_train=args.n_train,
                                                                                n_test=args.n_test, n_eval=args.n_eval,
                                                                                data_dim=args.data_dim,
                                                                                manywell_data_dim=args.manywell_dim)
        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_data_dim_{args.manywell_total_data_dim}"
                              f"_manywell_dim_{args.manywell_dim}"
                              f"_n_train_test_{args.n_train_test}_n_eval_{args.n_eval}"
                              f"_timestamp_{TIMESTAMP}.pkl")


    elif dataset_name == "gaussian_mixture_2d":
        data_dim = 2
        has_arg = hasattr(args, "gm_2d_num_components") and hasattr(args,"gm_2d_radius") and hasattr(args,"gm_2d_variance")
        n_comp = args.gm_2d_num_components
        data_snapshot_dict = (DataSnapShotUtils.
                              generate_transport_data_snapshot(dataset_name=dataset_name,
                                                               x_dim=data_dim, y_dim=data_dim,
                                                               n_train=args.n_train,
                                                               n_test=args.n_test, n_eval=args.n_eval,
                                                               num_components=n_comp,
                                                               radius = args.gm_2d_radius,
                                                               variance = args.gm_2d_variance))
        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_ncomp_{n_comp}"
                              f"_radius_{args.gm_2d_radius}"
                              f"_variance_{args.gm_2d_variance}"
                              f"_data_dim_{data_dim}"
                              f"_n_train_test_eval_{args.n_train}_{args.n_test}_{args.n_eval}.pkl")
    elif dataset_name == "skewed_gaussian_2d":
        dist_params = ["mean", "cov", "shape"]
        assert all(hasattr(args, dist_param) for dist_param in dist_params)
        x_dim = 2
        y_dim = 2
        kwargs = {"mean": args.mean, "cov": args.cov, "shape": args.shape}
        data_snapshot_dict = DataSnapShotUtils.generate_transport_data_snapshot(dataset_name=dataset_name,
                                                                                x_dim=x_dim, y_dim=y_dim,
                                                                                n_train=args.n_train,
                                                                                n_test=args.n_test, n_eval=args.n_eval,
                                                                                **kwargs)
        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_mean_{str(kwargs["mean"])}"
                              f"_cov_{str(kwargs["cov"])}"
                              f"_shape_{kwargs["shape"]}"
                              f"_n_train_{args.n_train}"
                              f"_n_test_{args.n_test}"
                              f"_n_eval_{args.n_eval}.pkl")

    # 2d sklearn 2d distributions
    elif dataset_name in ["circles"]:
        x_dim = y_dim = 2
        data_snapshot_dict = (DataSnapShotUtils.
                              generate_transport_data_snapshot(dataset_name=dataset_name,
                                                                                x_dim=x_dim, y_dim=y_dim,
                                                                                n_train=args.n_train,
                                                                                n_test=args.n_test, n_eval=args.n_eval))
        Visualizer.plot_one_2d_distribution(x=data_snapshot_dict["train"]["x1"],
                                            save_path=os.path.join(DATASET_FIGURES_PATH,f"{dataset_name}_{TIMESTAMP}.png"))
        snapshot_file_name = (f"data_snapshot_{dataset_name}"
                              f"_n_train_{args.n_train}"
                              f"_n_test_{args.n_test}"
                              f"_n_eval_{args.n_eval}.pkl")

    else:
        raise ValueError(f"dataset_name {dataset_name} not supported")

    logger.info(f"Saving snapshot file...")
    snapshot_filepath = os.path.join(DATASETS_SNAPSHOTS_DIR, snapshot_file_name)
    with open(snapshot_filepath, "wb") as f:
        pickle.dump(data_snapshot_dict, f)
        f.flush()
        f.close()
    logger.info(f"Successfully saved snapshot file to path = {snapshot_filepath}")
