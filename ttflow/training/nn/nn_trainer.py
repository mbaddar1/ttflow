"""
This script is a clean sheet for this one
flow_matching/sandbox/DMRG/tt_recflow_param_sharing_arch_models_trainer_old.py
to make stuff more modular and testable

Reference Implementation of RecFlow (NN version)
https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P?usp=sharing
https://github.com/gnobitab/RectifiedFlow#interactive-colab-notebooks (other versions for v impl.)

Sample runs

"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

from ttflow.training.nn.nn_models import VelocityFieldRegressionNet
from ttflow.training.nn.nn_trainer_utils import NeuralNetworkTrainer
from ttflow.utils import check_system_memory_and_swap, get_inmemory_size_mb

# Constants
MIN_MEM_GB = 30
MIN_SWAP_GB = 32
SEED = 42
# logging
TIMESTAMP = datetime.now().isoformat()
LOGS_PATH = "../logs"
EXPERIMENT_LOGFILE_NAME = f"vt_param_share_arch_{TIMESTAMP}.log"
EXPERIMENT_LOGFILE_PATH = os.path.join(LOGS_PATH, EXPERIMENT_LOGFILE_NAME)

logger.add(
    EXPERIMENT_LOGFILE_PATH,
    rotation="500 MB",  # Rotate when the file reaches 500 MB (or "1 day", "12:00", etc.)
    retention="10 days",  # Automatically clean up files older than 10 days (or count: 5)
    compression="zip",  # Compress rotated files to save space ("zip", "tar.gz")
    level="DEBUG",  # Minimum log level for this sink
    encoding="utf-8",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run with a data snapshot.")
    parser.add_argument(
        "--data-snapshot-filepath",
        type=str,
        help="Path to the .pkl data snapshot file.",
    )
    parser.add_argument(
        "--use-gpu",
        default=True,
        action="store_true",
        help="Use the GPU if available (default: CPU).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",  # defaults to False; becomes True when the flag is passed
        help="Show what would happen without making changes",
    )
    parser.add_argument("--output-model-dir", default="models/experimental", required=False)
    return parser.parse_args()

if __name__ == "__main__":
    # pars args
    args = parse_args()
    logger.info(f"args = {args}")
    # device and dtype
    data_type = torch.float32
    assert data_type == torch.float32
    is_available = torch.cuda.is_available()
    logger.info(f"is GPU/CUDA available = {is_available}")
    # set device
    device = torch.device("cuda" if is_available and args.use_gpu else "cpu")
    logger.info(f"device = {device}")
    # NN arch and train params
    nn_hidden_dim = 256  # TODO , I am using 512 in this arch, while in the time-grid arch I am using 128 - investigate later
    nn_depth = 6
    nn_epochs = 1 if args.dry_run else 200  # TODO : I am using 100 here
    lr = 1e-3
    batch_size = 256
    nn_reg_lambda = 1e-3
    test_ratio = 0.2
    patience = 10
    min_rel_loss_delta = 5e-5
    # load file
    data_snapshot_filepath = args.data_snapshot_filepath
    logger.info(f"Using data snapshot: {data_snapshot_filepath}")
    # check params
    assert hasattr(args,
                   "data_snapshot_filepath") and args.data_snapshot_filepath is not None, "data_snapshot_filepath is None"
    # checking memory
    assert check_system_memory_and_swap(min_ram_gb=MIN_MEM_GB, min_swap_gb=MIN_SWAP_GB), "low memory, increase! "
    # loads data
    with open(data_snapshot_filepath, "rb") as f:
        data_snapshot = pickle.load(f)
    # data metadata
    dataset_name = data_snapshot["dataset_name"]
    # monitoring data size
    data_pkl_mem_size_mb = get_inmemory_size_mb(obj=data_snapshot)
    logger.info(f"data pkl size = {np.round(data_pkl_mem_size_mb / 1000.0, 2)} GB")

    # ( 1 ) Starting RegNN training
    # load training data
    x = data_snapshot["train"]["l2"]["space-time"]["x"]
    t = data_snapshot["train"]["l2"]["space-time"]["t"]
    x_aug = torch.cat([x, t], dim=1)
    y = data_snapshot["train"]["l2"]["space-time"]["y"]
    x_aug = x_aug.to(device, data_type)
    y = y.to(device, data_type)

    logger.info(f"x_aug shape = {x_aug.shape}, y shape = {y.shape}")
    x_aug_train, x_aug_test, y_train, y_test = train_test_split(
        x_aug, y, test_size=test_ratio, random_state=SEED, shuffle=True
    )
    logger.info(
        f"x_aug_train/test y_train/test sizes = {x_aug_train.shape}, {x_aug_test.shape}, {y_train.shape}, "
        f"{y_test.shape}")
    # x_test =
    # model arch
    nn_model = VelocityFieldRegressionNet(input_dim=x_aug_train.shape[1], hidden_dim=nn_hidden_dim,
                                          output_dim=y_train.shape[1], depth=nn_depth, device=device,
                                          data_type=data_type)
    start_time = datetime.now()
    nn_learning_curve = NeuralNetworkTrainer.run_base_line(model=nn_model, x_train=x_aug_train,
                                                           y_train=y_train,
                                                           x_test=x_aug_test, y_test=y_test, batch_size=1024, lr=1e-4,
                                                           epochs=nn_epochs,
                                                           lambd=1e-3, min_rel_delta_loss=min_rel_loss_delta,
                                                           patience=patience)

    end_time = datetime.now()
    output_model_artifact = {"data_snapshot_filepath": data_snapshot_filepath, "model":
        nn_model, "model_type": "nn", "epochs": nn_epochs, "model_depth": nn_depth,
                             "batch_size": batch_size, "lr": lr,
                             "reg_lambda": nn_reg_lambda}
    data_params = "_".join(f"{k}_{v}" for k, v in data_snapshot["params"].items())
    output_model_filename = f"vt_nn_spacetime_{dataset_name}_data_params_{data_params}_nn_hidden_dim_{nn_hidden_dim}_{TIMESTAMP}.pkl"
    output_model_artifact_path = os.path.join(args.output_model_dir, output_model_filename)
    with open(output_model_artifact_path, "wb") as f:
        pickle.dump(output_model_artifact, f)
        f.flush()
        f.close()
    logger.info("Successfully saved model to path : {}".format(output_model_artifact_path))
