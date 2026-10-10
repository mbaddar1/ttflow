"""
This script is a clean sheet for this one
flow_matching/sandbox/DMRG/tt_recflow_param_sharing_arch_models_trainer_old.py
to make stuff more modular and testable

Reference Implementation of RecFlow (NN version)
https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P?usp=sharing
https://github.com/gnobitab/RectifiedFlow#interactive-colab-notebooks (other versions for v impl.)

Sample runs
(cmd-args)
( 1 )
--data-snapshot-filepath
"data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_500000_200000.pkl"
--vt-model
"nn"
--vt-opt
"adam"
--trajectory-model
"global"

Reference Impl for RecFlow
https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P?usp=sharing

"""

import argparse
import math
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split, learning_curve

from ttflow.data_generation.data_utils import DataUtils
from ttflow.training.nn.nn_models import VelocityFieldRegressionNet
from ttflow.training.nn.nn_trainer_utils import NeuralNetworkTrainingUtils
from ttflow.utils import check_system_memory_and_swap, get_inmemory_size_mb

# Constants
MIN_MEM_GB = 30
MIN_SWAP_GB = 32
SEED = 42

# FILE PATHS
# param choices
VT_MODEL_CHOICES = ["nn", "tt"]
VT_OPT_CHOICES = ["adam",  # nn
                  "als", "dmrg-truncation", "dmrg-entropy"  # tt
                  ]
TRAJECTORY_MODELING_CHOICES = ["global", "piecewise"]


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
    parser.add_argument("--vt-model", type=str, choices=VT_MODEL_CHOICES,
                        help="modeling vt either bt nn or tt", required=True)
    parser.add_argument("--vt-opt", type=str, choices=VT_OPT_CHOICES,
                        help="opt for vt model", required=True)
    parser.add_argument("--trajectory-model", type=str, choices=TRAJECTORY_MODELING_CHOICES,
                        help="either modeling the global trajectory or on piecewise basis ", required=True)
    parser.add_argument("--rectification-linearity", type=str, choices=["linear", "nonlinear"],
                        default="linear", help="Is the rectification process linear or non-linear")
    parser.add_argument(
        "--dry-run",
        action="store_true",  # defaults to False; becomes True when the flag is passed
        help="Show what would happen without making changes",
    )
    # dirs
    parser.add_argument("--logs-dir", type=str, help="Path to the logs directory", default="logs")
    parser.add_argument("--output-model-dir", default="models/experimental", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    # pars args
    args = parse_args()
    # logging metadata , dependent on args
    # logging
    experiment_timestamp = datetime.now().isoformat()

    experiment_log_filename = (f"vt_model_{args.vt_model}_opt_{args.vt_opt}_time_model_{args.trajectory_model}_"
                               f"timestamp_{experiment_timestamp}.log")
    experiments_log_filepath = os.path.join(args.logs_dir, experiment_log_filename)
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
    # same arch and opt hyperparams in
    #   https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P?usp=sharing
    nn_total_num_param_updates = 10_000 + 1  # number of updates for each sample
    nn_hidden_dim = 128
    batch_size = 2048
    nn_depth = 2
    lr = 1e-3
    nn_reg_lambda = 1e-3
    patience = 10
    min_rel_loss_delta = 5e-5
    # trajectory params
    # see https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html#nonlinear-rectified-flows
    # load file
    data_snapshot_filepath = args.data_snapshot_filepath
    logger.info(f"Using data snapshot: {data_snapshot_filepath}")
    # check params
    assert hasattr(args,
                   "data_snapshot_filepath") and args.data_snapshot_filepath is not None, "data_snapshot_filepath is None"
    assert hasattr(args, "trajectory_model") and args.trajectory_model is not None, f"args.trajectory_model is None"
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
    # data params text  - to be used in output file name
    data_params = "_".join(f"{k}_{v}" for k, v in data_snapshot["params"].items())
    # load x0,x1 for train/test splits (no eval , yet)
    x0_train = data_snapshot["train"]["x0"].to(dtype=data_type, device=device)
    x1_train = data_snapshot["train"]["x1"].to(dtype=data_type, device=device)
    x0_test = data_snapshot["test"]["x0"].to(dtype=data_type, device=device)
    x1_test = data_snapshot["test"]["x1"].to(dtype=data_type, device=device)
    # (dynamically) determine epochs - fixing the number of param updates and determine nn_epochs based on it
    n_train = x0_train.shape[0]
    num_batches_per_epoch = math.ceil(float(n_train) / batch_size)
    nn_epochs = int(math.ceil(nn_total_num_param_updates / num_batches_per_epoch))
    nn_epochs = 10 * math.ceil(nn_epochs / 10) * 2  # approximate to closest 10 then double
    #
    if args.trajectory_model == "global":
        # generate xt (train,test)
        logger.info(f"Generating training data for the global trajectory training process..")
        xt_train, t_train, y_train = DataUtils.get_rectified_flow_get_l2_data(z0=x0_train, z1=x1_train,
                                                                              zt_mode=args.rectification_linearity)
        xt_test, t_test, y_test = DataUtils.get_rectified_flow_get_l2_data(z0=x0_test, z1=x1_test,
                                                                           zt_mode=args.rectification_linearity)
        # augment
        xt_train_aug = torch.cat([xt_train, t_train], dim=1)
        xt_test_aug = torch.cat([xt_test, t_test], dim=1)
        logger.info(f"Generating training data with xt_train_aug.shape = {xt_train_aug.shape} "
                    f"x_test_aug.shape = {xt_test_aug.shape} "
                    f"y_train.shape = {y_train.shape} "
                    f"y_test.shape = {y_test.shape}")

        if args.vt_model == "nn":
            # create RegNet
            nn_model = VelocityFieldRegressionNet(
                input_dim=xt_train_aug.shape[1],
                hidden_dim=nn_hidden_dim,
                output_dim=y_train.shape[1],
                depth=nn_depth,
                device=device,
                data_type=data_type,
            )
            if args.vt_opt == "adam":
                # create adam object
                # call Vt nn trainer
                nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=nn_reg_lambda)
                logger.info(f"Created optimizer : {nn_optimizer}")
                logger.info(
                    f"Training model: {nn_model} | Optimizer: {nn_optimizer} | "
                    f"Train shape: {xt_train_aug.shape} | Test shape: {xt_test_aug.shape} | "
                    f"Batch size: {batch_size} | Epochs: {nn_epochs} | "
                    f"Patience: {patience} | Min rel loss delta: {min_rel_loss_delta}"
                )
                learning_curve = NeuralNetworkTrainingUtils.train_nn(model=nn_model, optimizer=nn_optimizer,
                                                                     x_train=xt_train_aug, y_train=y_train,
                                                                     x_test=xt_test_aug, y_test=y_test,
                                                                     min_rel_loss_delta=min_rel_loss_delta,
                                                                     batch_size=batch_size,
                                                                     epochs=nn_epochs,
                                                                     patience=patience)
                vt_artifact = {"model": nn_model,
                               "hyperparams": {"hidden_dim": nn_hidden_dim, "depth": nn_depth, "lr": lr,
                                               "batch_size": batch_size, "epochs": nn_epochs},
                               "data_snapshot_filepath": data_snapshot_filepath,
                               "trajectory_model": args.trajectory_model, "learning_curve": learning_curve,
                               "arch": "nn", "opt": args.vt_opt}
                model_hyperparams_text = "_".join(["hidden_dim", str(nn_hidden_dim), "depth", str(nn_depth)])

            else:
                raise ValueError(f"vt_opt = {args.vt_opt} not supported.")
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    output_model_filename = (f"vt_"
                             f"dataset_{dataset_name}_"
                             f"data_params_{data_params}_"
                             f"trajectory_model_{args.trajectory_model}_"
                             f"func_model_{args.vt_model}_"
                             f"model_hyperparams_{model_hyperparams_text}_"
                             f"opt_{args.vt_opt}_"
                             f""
                             f"nn_hidden_dim_{nn_hidden_dim}_timestamp_{experiment_timestamp}.pkl")
    output_model_artifact_path = os.path.join(args.output_model_dir, output_model_filename)
    with open(output_model_artifact_path, "wb") as f:
        pickle.dump(vt_artifact, f)
        f.flush()
        f.close()
    logger.info(f"Output model artifact saved to = {output_model_artifact_path}")
    logger.info(f"vt training process finished successfully.")
