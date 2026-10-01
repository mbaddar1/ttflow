import argparse
import os.path
import pickle
from typing import Tuple, Optional, Iterable
from loguru import logger
import torch.nn
from torchdiffeq import odeint
from datetime import datetime
from geomloss import SamplesLoss
from ttflow.viz.viz import Visualizer
from tqdm import tqdm
import numpy as np

INFERENCE_FIGURES_DIR = "figures/inference/experimental"
INFERENCE_LOGS_DIR = "logs/inference"

class InferenceHandler:
    def __init__(self, x_shape: Tuple[int, int], model: torch.nn.Module):
        self.x_shape = x_shape
        self.model = model

    @torch.no_grad()
    def infer(self, x0: torch.Tensor, t_eval: torch.Tensor,
              rtol: float = 1e-7,
              atol: float = 1e-9, method: Optional[str] = None) -> Iterable[torch.Tensor]:
        x0_flat = x0.ravel()
        traj = odeint(func=self.__fnn, y0=x0_flat, t=t_eval, method=method, rtol=rtol, atol=atol)
        return traj

    def __fnn(self, t: float, x: torch.Tensor) -> float:
        x_reshaped = x.reshape(self.x_shape)
        t_tensor = torch.ones(self.x_shape[0], 1).type(x.dtype).to(x.device) * t
        x_aug = torch.cat([x_reshaped, t_tensor], dim=1)
        output = self.model(x_aug)
        return output.ravel()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference or processing with VT artifact"
    )
    parser.add_argument(
        "--vt-artifact-file-path",
        type=str,
        help="Path to the VT artifact pickle file.",
        required=True
    )
    parser.add_argument(
        "--n-inference",
        type=int,
        help="Number of samples to use in the inference process",
        required=False
    )
    parser.add_argument(
        "--n-time-points",
        type=int,
        default=1000,
        help="Number time points for ODESOLVE (inference/generation process)",
        required=False
    )
    parser.add_argument(
        "--odesolve-method",
        type=str,
        choices=["rk4"],# FIXME add more
        default="rk4",
        help="ODESOLVE method for inference",
        required=False
    )
    return parser.parse_args()


TIMESTAMP = datetime.now().isoformat()
if __name__ == "__main__":
    # TODO Split inference / eval / viz to separate pipelines
    # load data snapshot and velocity model
    # do inference
    # plot inferred vs reference samples
    # calculate sink-horn score
    args = parse_args()
    logger.info(f"Parsed args = {args}")
    # set filepaths

    vt_artifact_file_path = args.vt_artifact_file_path
    # load
    logger.info("Loading vt_artifact from {}".format(vt_artifact_file_path))
    with open(vt_artifact_file_path, "rb") as f:
        vt_artifact = pickle.load(f)
    assert isinstance(vt_artifact, dict)
    data_snapshot_file_path = vt_artifact["data_snapshot_filepath"]
    logger.info("Loading data_snapshot from {}".format(data_snapshot_file_path))
    with open(data_snapshot_file_path, "rb") as f:
        data_snapshot = pickle.load(f)
    logger.info(f"Successfully loaded data_snapshot from {data_snapshot_file_path}")
    assert isinstance(data_snapshot, dict)
    dataset_name = data_snapshot["dataset_name"]
    data_params = data_snapshot["params"]
    data_params_flat = "_".join(f"{k}_{v}" for k, v in data_params.items())
    # assertions
    assert isinstance(data_snapshot, dict)
    assert isinstance(vt_artifact, dict)

    logger.info("Successfully loaded vt_artifact and the linked data snapshot")
    # do inference
    x0 = data_snapshot["eval"]["inference"]["x0"]
    x1 = data_snapshot["eval"]["inference"]["x1"]

    for j in range(len(x0.shape)):
        assert x0.shape[j] == x1.shape[j]
    logger.info(f"loaded x0 and x1, x0.shape = {x0.shape}, x1.shape = {x1.shape}")
    if hasattr(args, "n_inference") and isinstance(args.n_inference, int):
        logger.info(f"Passed args.n_inference = {args.n_inference}, using it...")
        n_inference = args.n_inference
        x0 = x0[:n_inference, :]
        x1 = x1[:n_inference, :]
    else:
        logger.info(f"A valid n_inference is not passed, using all x0 samples, x0.shape = {x0.shape}")
    vt_nn_model = vt_artifact["model"]
    assert isinstance(vt_nn_model, torch.nn.Module)
    param = next(vt_nn_model.parameters())
    model_device = param.device
    model_dtype = param.dtype
    x0 = x0.to(dtype=model_dtype, device=model_device)
    x1 = x1.to(dtype=model_dtype, device=model_device)
    odesolve_method = args.odesolve_method
    inference_handler = InferenceHandler(x_shape=x0.shape, model=vt_nn_model)
    t_eval = torch.linspace(start=0, end=1, steps=args.n_time_points).to(dtype=model_dtype, device=model_device)
    logger.info(
        f"Starting inference x0->x1_hat over t_eval : x0.shape = {x0.shape} | len(t_eval) = {len(t_eval)} | odesolve_method = {odesolve_method}")
    start_time = datetime.now()
    traj = inference_handler.infer(x0=x0, t_eval=t_eval, method=odesolve_method)
    end_time = datetime.now()
    inference_time_seconds = (end_time - start_time).total_seconds()
    logger.info(f"Inference time = {inference_time_seconds} seconds")
    x1_hat = list(traj)[-1].reshape(x1.shape)

    logger.info("Inference and evaluation process finished successfully")
    # sinkhorn loss
    sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.1)

    logger.info(f"Sinkhorn distance (x1,x1)= {sinkhorn_loss_fn(x1, x1).item()}")
    logger.info(f"Sinkhorn distance (x0,x0)= {sinkhorn_loss_fn(x0, x0).item()}")
    logger.info(f"Sinkhorn distance (x0,x1)= {sinkhorn_loss_fn(x0, x1).item()}")

    num_sinkhorn_iterations = 10
    sinkhorn_distance_values = []
    logger.info(f"Calculating sinkhorn-distance over {num_sinkhorn_iterations} iterations.")
    for iteration in tqdm(range(num_sinkhorn_iterations), desc="sinkhorn-calculations"):
        sinkhorn_loss_value = sinkhorn_loss_fn(x1_hat, x1).item()
        sinkhorn_distance_values.append(sinkhorn_loss_value)
    logger.info(f"Sinkhorn(x1_hat,x1) summary : mean = {np.mean(sinkhorn_distance_values)},"
                f"std = {np.std(sinkhorn_distance_values)}")
    # TODO make dataset specific visualization
    figure_path = os.path.join(INFERENCE_FIGURES_DIR, f"{dataset_name}_data_params_{data_params_flat}_{TIMESTAMP}.png")
    if dataset_name=="skewed_gaussian_2d":
        # TODO add model arch and trajectory modeling approach (grid , continous/space-time)
        Visualizer.plot_with_subplots_gaussian_contour_2d(x0=x0, x1=x1, x1_hat=x1_hat, title="Skewed Gaussian Distribution: Target vs. Inferred Samples", save_path=figure_path)
    elif dataset_name=="gaussian_mixture_2d":
        Visualizer.plot_with_subplots_gaussian_mixture_2d(x0=x0, x1=x1, x1_hat=x1_hat, title="Gaussian Mixture : Target vs. Inferred Samples ", save_path=figure_path)
    else:
        raise NotImplementedError("dataset_name has not supported viz")
    #
    # Visualizer.plot_three_2d_distributions(x0=x0, x1=x1, x2=x1_hat, label0="init", label1="reference",
    #                                        label2="inferred",
    #                                        title=f"Distribution comparison : {dataset_name}_{str(data_params)}",
    #                                        output_file_name=os.path.join(INFERENCE_FIGURES_DIR,
    #                                                                      f"{dataset_name}_data_params_{data_params_flat}_{TIMESTAMP}.png"))