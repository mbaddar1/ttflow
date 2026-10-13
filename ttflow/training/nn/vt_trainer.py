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
from datetime import datetime
from typing import Tuple, Iterable

from tqdm import tqdm
import numpy as np
import torch
from loguru import logger

from ttflow.data_generation.data_utils import DataUtils
from ttflow.inference.inference_and_evaluation import InferenceHandler
from ttflow.training.nn.nn_models import VelocityFieldRegressionNet
from ttflow.training.nn.nn_trainer_utils import NeuralNetworkTrainingUtils
from ttflow.training.nn.rectified_flows_utils import RectifiedFlowsUtils
from ttflow.training.nn.vt_trainer_utils import VTTrainerUtils
from ttflow.utils import check_system_memory_and_swap, get_inmemory_size_mb
from datetime import datetime

RUN_TIMESTAMP = datetime.now().isoformat()
#
LOGGER_DIR = "logs/training"

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
    # rectified flow modeling parameters
    parser.add_argument("--vt-model", type=str, choices=VT_MODEL_CHOICES,
                        help="modeling vt either nn or tt", required=True)
    parser.add_argument("--vt-opt", type=str, choices=VT_OPT_CHOICES,
                        help="opt for vt model", required=True)
    parser.add_argument("--trajectory-model", type=str, choices=TRAJECTORY_MODELING_CHOICES,
                        help="either modeling the global trajectory or on piecewise basis ", required=True)
    parser.add_argument("--odesolve-method", type=str, choices=["rk4"],
                        help="odesolve method", required=False)
    parser.add_argument("--rectification-linearity", type=str, choices=["linear", "nonlinear"],
                        default="linear", help="Is the rectification process linear or non-linear")
    parser.add_argument("--reflow-iterations", type=int, default=1, help="Number of reflows")
    # NN arch and train params
    # same arch and opt hyperparams in
    #   https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P?usp=sharing

    # trajectory params
    # see https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html#nonlinear-rectified-flows

    parser.add_argument(
        "--nn-epochs",
        type=int,
        default=200,  # for range starting form zero
        help="number of epochs",
    )
    parser.add_argument(
        "--nn-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size of the neural network (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training (default: 2048)",
    )
    parser.add_argument(
        "--nn-depth",
        type=int,
        default=2,
        help="Number of layers in the neural network (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--nn-reg-lambda",
        type=float,
        default=1e-3,
        help="L2 regularization weight lambda (default: 0.001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience steps for early stopping (default: 10)",
    )
    parser.add_argument(
        "--min-rel-loss-delta",
        type=float,
        default=5e-5,
        help="Minimum relative loss reduction required to count as an improvement (default: 5e-05)",
    )
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

    # set nn_epochs
    nn_epochs = 1 if args.dry_run else args.nn_epochs
    logger.info(f"args = {args}")
    # device and dtype
    data_type = torch.float32
    assert data_type == torch.float32
    is_available = torch.cuda.is_available()
    logger.info(f"is GPU/CUDA available = {is_available}")
    # set device
    device = torch.device("cuda" if is_available and args.use_gpu else "cpu")
    logger.info(f"device = {device}")
    # inference for reflow
    t_eval = torch.linspace(0, 1, 1000).to(dtype=data_type, device=device)
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
    # log path
    log_file_path = os.path.join(LOGGER_DIR, f"vt_trainer_"
                                             f"{args.vt_model}_"
                                             f"{args.vt_opt}_"
                                             f"{args.trajectory_model}_"
                                             f"{RUN_TIMESTAMP}.log")
    logger.add(
        log_file_path,
        rotation="10 MB",  # Rotate when file size reaches 10MB (or "1 day", "00:00", etc.)
        retention="10 days",  # Keep old log files for 10 days before deleting
        compression="zip",  # Compress rotated files into .zip
        level="DEBUG",  # Minimum severity level to log to this file
        enqueue=True,  # Thread/process-safe logging
        encoding="utf-8",
    )
    # monitoring data size
    data_pkl_mem_size_mb = get_inmemory_size_mb(obj=data_snapshot)
    logger.info(f"data pkl size = {np.round(data_pkl_mem_size_mb / 1000.0, 2)} GB")
    # data params text  - to be used in output file name
    data_params = "_".join(f"{k}_{v}" for k, v in data_snapshot["params"].items())
    # load x0,x1 for train/test splits (no eval , yet)
    z1_train = data_snapshot["train"]["x1"].to(dtype=data_type, device=device)
    z1_test = data_snapshot["test"]["x1"].to(dtype=data_type, device=device)
    # (dynamically) determine epochs - fixing the number of param updates and determine nn_epochs based on it
    n_train = z1_train.shape[0]
    num_batches_per_epoch = math.ceil(float(n_train) / args.batch_size)
    if args.trajectory_model == "global":
        # generate xt (train,test)
        logger.info(f"Generating training data for the global trajectory training process..")
        if args.vt_model == "nn":
            # create RegNet
            if args.vt_opt == "adam":

                z0_train = DataUtils.generate_gaussian_prior(dim=z1_train.shape[1],
                                                             n_samples=z1_train.shape[0]).to(dtype=data_type,
                                                                                             device=device)
                z0_test = DataUtils.generate_gaussian_prior(dim=z1_test.shape[1],
                                                            n_samples=z1_test.shape[0]).to(dtype=data_type,
                                                                                           device=device)

                nn_models = []  # TODO - for tracking only, always use the last model in global reflow
                learning_curves = []  # for each reflowed model
                # train on z0,z1 (train and test)
                nn_model = VelocityFieldRegressionNet(
                    input_dim=z0_train.shape[1] + 1,
                    hidden_dim=args.nn_hidden_dim,
                    output_dim=z0_train.shape[1],
                    depth=args.nn_depth,
                    device=device,
                    data_type=data_type,
                )

                nn_optimizer = torch.optim.Adam(
                    nn_model.parameters(), lr=args.lr, weight_decay=args.nn_reg_lambda
                )
                logger.info(
                    "***********************************************************"
                    "Training Model Specs:\n"
                    f"  [Model Architecture]\n"
                    f"    - model =                   {nn_model}\n"
                    f"    - Input Dim:                {z0_train.shape[1] + 1}\n"
                    f"    - Hidden Dim:               {args.nn_hidden_dim}\n"
                    f"    - Output Dim:               {z0_train.shape[1]}\n"
                    f"    - Depth:                    {args.nn_depth}\n"
                    f"    - Device:                   {device}\n"
                    f"    - Data Type:                {data_type}\n"
                    f"  [Optimizer Specs]\n"
                    f"    - Optimizer :               {nn_optimizer}:\n"
                    f"    - Learning Rate:            {args.lr}\n"
                    f"    - Weight Decay (lambda):    {args.nn_reg_lambda}\n"
                    f"  [Training Params]\n"
                    f"    - Epochs:                   {nn_epochs}\n"
                    f"    - Batch Size:               {args.batch_size}\n"
                    f"    - Patience:                 {args.patience}\n"
                    f"    - Min Rel Loss Delta:       {args.min_rel_loss_delta}\n"
                    f"    - ZT Mode:                  {args.rectification_linearity}\n"
                    f"    - Reflow Iterations:  {args.reflow_iterations}"
                    "**********************************************************"
                )
                for reflow_iteration in tqdm(range(args.reflow_iterations + 1), desc="reflow iterations"):
                    logger.info("Reflow iteration: {}".format(reflow_iteration))
                    learning_curve = VTTrainerUtils.reflow_nn_train(model=nn_model, optimizer=nn_optimizer,
                                                                    z0_train=z0_train, z1_train=z1_train,
                                                                    z0_test=z0_test, z1_test=z1_test,
                                                                    batch_size=args.batch_size,
                                                                    epochs=nn_epochs, patience=args.patience,
                                                                    min_rel_loss_delta=args.min_rel_loss_delta,
                                                                    zt_mode=args.rectification_linearity)
                    zt_test, t_test, y_test = DataUtils.get_rectified_flow_get_l2_data(z0=z0_test, z1=z1_test,
                                                                                       zt_mode=args.rectification_linearity)
                    zt_test_aug = torch.cat([zt_test, t_test], dim=1)
                    with torch.no_grad():
                        nn_model.eval()
                        y_pred = nn_model(zt_test_aug)
                        nn_models.append(nn_model)
                    if reflow_iteration == args.reflow_iterations:
                        logger.info(f"Given num_reflow_iterations = {args.reflow_iterations} "
                                    f"and current reflow_iteration = {reflow_iteration}, "
                                    f"no further reflows are needed, exiting the reflow loop")
                        break
                    # reflow part
                    # z0_train,z0_test (new reflow iteration)
                    inference_batch_size = 1024 * 4
                    logger.info(f"Reflow iteration # {reflow_iteration} : "
                                f"Generating z0_train,z1_train for reflow iteration # {reflow_iteration + 1}")
                    z0_train, z1_train = RectifiedFlowsUtils.generate_reflow_data(n=z0_train.shape[0],
                                                                                  dim=z0_train.shape[1],
                                                                                  t_eval=t_eval,
                                                                                  vt_model=nn_model,
                                                                                  data_type=data_type,
                                                                                  device=device,
                                                                                  inference_batch_size=inference_batch_size,
                                                                                  ode_solve_method=args.odesolve_method)
                    # z0_test,z1_test (new reflow iteration)
                    logger.info(
                        f"Reflow iteration # {reflow_iteration + 1} : "
                        f"Generating z0_test,z1_test for reflow_iteration # {reflow_iteration + 2}")
                    z0_test, z1_test = RectifiedFlowsUtils.generate_reflow_data(n=z0_test.shape[0],
                                                                                dim=z0_test.shape[1],
                                                                                t_eval=t_eval,
                                                                                vt_model=nn_model,
                                                                                data_type=data_type,
                                                                                device=device,
                                                                                inference_batch_size=inference_batch_size,
                                                                                ode_solve_method=args.odesolve_method)
                vt_artifact = {"models": nn_models, "reflows": args.reflow_iterations,
                               "hyperparams": {"hidden_dim": args.nn_hidden_dim, "depth": args.nn_depth,
                                               "lr": args.lr,
                                               "batch_size": args.batch_size, "epochs": args.nn_epochs},
                               "data_snapshot_filepath": data_snapshot_filepath,
                               "trajectory_model": args.trajectory_model, "learning_curves": learning_curves,
                               "arch": "nn", "opt": args.vt_opt}
                model_hyperparams_text = "_".join(["hidden_dim", str(args.nn_hidden_dim), "depth", str(args.nn_depth)])

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
                             f"nn_hidden_dim_{args.nn_hidden_dim}_timestamp_{experiment_timestamp}.pkl")
    output_model_artifact_path = os.path.join(args.output_model_dir, output_model_filename)
    with open(output_model_artifact_path, "wb") as f:
        pickle.dump(vt_artifact, f)
        f.flush()
        f.close()
    logger.info(f"Output model artifact saved to = {output_model_artifact_path}")
    logger.info(f"vt training process finished successfully.")
