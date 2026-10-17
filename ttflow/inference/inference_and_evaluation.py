import argparse
import os.path
import pickle
from typing import Tuple, Optional, Iterable, List, Union
from loguru import logger
import torch.nn
from torchdiffeq import odeint
from datetime import datetime
from geomloss import SamplesLoss

from ttflow.data_generation.data_utils import DataUtils
from ttflow.viz.viz import Visualizer
from tqdm import tqdm
import numpy as np

INFERENCE_FIGURES_DIR = "figures/inference/experimental"
INFERENCE_LOGS_DIR = "logs/inference"


class InferenceHandler:
    INFERENCE_MODES = ["one-trajectory", "reflow", "piecewise"]

    def __init__(self, models: Union[List[torch.nn.Module] | torch.nn.Module],
                 inference_mode: str):
        self.x_shape = None
        self.pbar = None
        self.last_t = None
        if isinstance(models, torch.nn.Module):
            self.model = models
        elif isinstance(models, list) and all(isinstance(model, torch.nn.Module) for model in models):
            self.models = models
        else:
            raise ValueError(f"Invalid models parameters")

        self.inference_mode = inference_mode
        assert inference_mode in InferenceHandler.INFERENCE_MODES, "Inference mode must be one of {}".format(
            InferenceHandler.INFERENCE_MODES)
        if inference_mode == "one-trajectory":
            assert isinstance(models, torch.nn.Module)
        else:
            raise NotImplementedError(f"{self.inference_mode} not implemented yet!")

    @torch.no_grad()
    def infer(self, x0: torch.Tensor, t_eval: torch.Tensor,
              rtol: float = 1e-7,
              atol: float = 1e-9, ode_solver_method: Optional[str] = None) -> Iterable[torch.Tensor]:
        # TODO handle global and piecewise linear trajectory
        self.x_shape = x0.shape  # FIXME a work around
        self.pbar = tqdm(total=t_eval[-1].item() - t_eval[0].item(), desc=f"Solving ODE | Method = {ode_solver_method}")
        self.last_t = t_eval[0]
        if self.inference_mode == "one-trajectory":
            x0_flat = x0.ravel()
            traj = odeint(func=self.__fnn, y0=x0_flat, t=t_eval, method=ode_solver_method, rtol=rtol, atol=atol)
            return traj
        else:
            raise NotImplementedError(f"{self.inference_mode} not implemented yet!")

    def __fnn(self, t: float, x: torch.Tensor) -> float:
        assert self.x_shape is not None
        assert self.pbar is not None
        assert self.last_t is not None
        if self.inference_mode == "one-trajectory":
            x_reshaped = x.reshape(self.x_shape)
            t_tensor = torch.ones(self.x_shape[0], 1).type(x.dtype).to(x.device) * t
            x_aug = torch.cat([x_reshaped, t_tensor], dim=1)
            output = self.model(x_aug)
            # update pbar
            dt = (t - self.last_t).item()
            if dt > 0:
                self.pbar.update(dt)
                self.last_t = t
            return output.ravel()
        else:
            raise NotImplementedError(f"{self.inference_mode} not implemented yet!")


class TransportEvaluator:
    @staticmethod
    def evaluate_sinkhorn_by_blur(
            x,
            y,
            blurs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
            n_repeats=5,
            sample_size=10_000,
            scaling=0.99,
            seed=42,
    ):
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError("X and Y must have shape [N, D] with matching D.")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("X and Y must contain only finite values.")
        if not 2 <= sample_size <= min(len(x), len(y)):
            raise ValueError("Invalid sample_size.")
        if n_repeats < 2:
            raise ValueError("n_repeats must be at least 2.")

        # Estimate characteristic data scale from real samples.
        x_scale = x[:min(len(x), 5000)]
        with torch.no_grad():
            distances = torch.cdist(x_scale, x_scale)
            distances.fill_diagonal_(float("inf"))
            typical_distance = distances.min(dim=1).values.median().item()

        # Add data-scale-aware blur candidates.
        blurs = sorted(set(blurs) | {
            typical_distance * multiplier
            for multiplier in (0.1, 0.25, 0.5, 1.0, 2.0)
        })

        generator = torch.Generator(device="cpu").manual_seed(seed)
        results = {}

        for blur in tqdm(blurs, desc="sinkhorn-blurs"):
            values = {
                "real_generated": [],
                "real_real": [],
                "generated_generated": [],
            }

            loss_fn = SamplesLoss(
                loss="sinkhorn",
                p=2,
                blur=float(blur),
                debias=True,
                scaling=scaling,
                backend="online",
            )

            for _ in tqdm(range(n_repeats), desc="sinkhorn-repeats"):
                idx_x1 = torch.randperm(len(x), generator=generator)[:sample_size]
                idx_x2 = torch.randperm(len(x), generator=generator)[:sample_size]
                idx_y1 = torch.randperm(len(y), generator=generator)[:sample_size]
                idx_y2 = torch.randperm(len(y), generator=generator)[:sample_size]

                X1 = x[idx_x1.to(x.device)]
                X2 = x[idx_x2.to(x.device)]
                Y1 = y[idx_y1.to(y.device)]
                Y2 = y[idx_y2.to(y.device)]

                with torch.no_grad():
                    rg = loss_fn(X1, Y1).item()
                    rr = loss_fn(X1, X2).item()
                    gg = loss_fn(Y1, Y2).item()

                for name, value in (
                        ("real_generated", rg),
                        ("real_real", rr),
                        ("generated_generated", gg),
                ):
                    if not torch.isfinite(torch.tensor(value)):
                        raise RuntimeError(
                            f"Non-finite {name} loss for blur={blur}"
                        )
                    values[name].append(value)

            results[blur] = {
                name: {
                    "mean": float(torch.tensor(v, dtype=torch.float64).mean()),
                    "stddev": float(
                        torch.tensor(v, dtype=torch.float64).std(correction=1)
                    ),
                }
                for name, v in values.items()
            }

        return results


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
        "--ode-solver-method",
        type=str,
        choices=["rk4", "euler", "dopri8"],  # FIXME add more
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
    # set inference params

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
    # set inference hyperparams
    traj_model = vt_artifact["trajectory_model"]
    # mind the underscore in param name
    has_ode_solver_method_param = hasattr(args, "ode_solver_method") and args.ode_solver_method is not None
    if not has_ode_solver_method_param:
        raise ValueError("Must have ode-solver-method-param")
    ode_solver_method = args.ode_solver_method
    logger.info(f"traj_model = {traj_model}, ode_method = {ode_solver_method}")
    num_reflows = vt_artifact["reflows"]
    # do inference
    x1 = data_snapshot["eval"]["x1"]
    x0 = DataUtils.generate_gaussian_prior(n_samples=x1.shape[0], dim=x1.shape[1])

    logger.info(f"x0.shape = {x0.shape}, x1.shape = {x1.shape}")
    if hasattr(args, "n_inference") and isinstance(args.n_inference, int):
        logger.info(f"Passed args.n_inference = {args.n_inference}, using it...")
        n_inference = args.n_inference
        x0 = x0[:n_inference, :]
        x1 = x1[:n_inference, :]
    else:
        logger.info(f"A valid n_inference is not passed, using all x0 samples, x0.shape = {x0.shape}")
    if vt_artifact["trajectory_model"] == "global":
        vt_nn_model = vt_artifact["models"][
            -1]  # get the latest model, I keep all reflow model just for possible debugging purposes
        assert isinstance(vt_nn_model, torch.nn.Module)
        param = next(vt_nn_model.parameters())
        model_device = param.device
        model_dtype = param.dtype
        x0 = x0.to(dtype=model_dtype, device=model_device)
        x1 = x1.to(dtype=model_dtype, device=model_device)
        inference_handler = InferenceHandler(models=vt_nn_model, inference_mode="one-trajectory")
        t_eval = torch.linspace(start=0, end=1, steps=args.n_time_points).to(dtype=model_dtype, device=model_device)
        logger.info(
            f"*** Starting inference x0->x1_hat over t_eval : x0.shape = {x0.shape} | "
            f"len(t_eval) = {len(t_eval)} | "
            f"odesolve_method = {args.ode_solver_method}***")
        start_time = datetime.now()
        traj = inference_handler.infer(x0=x0, t_eval=t_eval, ode_solver_method=ode_solver_method)
        end_time = datetime.now()
        inference_time_seconds = (end_time - start_time).total_seconds()
    else:
        raise NotImplementedError("This inference case is not implemented")
    x1_hat = list(traj)[-1].reshape(x0.shape)
    # sinkhorn loss
    # https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html
    sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, debias=True, scaling=0.99)
    num_sinkhorn_iterations = 10
    sinkhorn_distance_values = []

    assert len(x1_hat.shape) == len(x1.shape)
    assert len(x1.shape) == 2
    for j in range(len(x1.shape)):
        assert x1.shape[j] == x1_hat.shape[j]
    logger.info("*** Starting the evaluation process ***")
    eval_result = TransportEvaluator.evaluate_sinkhorn_by_blur(x=x1, y=x1_hat)
    # n_subsample = x1.shape[0] // 4
    # logger.info(f"Bootstrap calculations for sinkhorn-distance over {num_sinkhorn_iterations} "
    #             f"iterations, with n_subsample = {n_subsample} out of n_samples = {x1.shape[0]}")
    # # sinkhorn calculations
    # for iteration in tqdm(range(num_sinkhorn_iterations), desc="sinkhorn-calculations"):
    #     i_idx = torch.randperm(x1.shape[0], device=x1.device)[:n_subsample]
    #     sinkhorn_loss_value = sinkhorn_loss_fn(x1_hat[i_idx, :], x1[i_idx, :]).item()
    #     logger.info(f"@Iteration # {(iteration + 1)}, sinkhorn_loss_value = {sinkhorn_loss_value}")
    #     sinkhorn_distance_values.append(sinkhorn_loss_value)

    SEP_LINE_LEN = 50
    HEADER_TEXT = "Sinkhorn Results Summary"

    logger.info(
        f"\n{'=' * SEP_LINE_LEN}\n"
        f"{HEADER_TEXT.center(SEP_LINE_LEN)}\n"
        f"{'=' * SEP_LINE_LEN}\n"
        f"  ODE Solver Method    : {ode_solver_method}\n"
        f"  VT Trajectory Model  : {traj_model}\n"
        f"  Num Reflows          : {num_reflows}\n"
        f"  Sinkhorn Iterations  : {num_sinkhorn_iterations}\n"
        f"  Inference Samples    : {x1.shape[0]:,}\n"
        f"  Inference Time       : {inference_time_seconds:.2f}s\n"
        f"  Evaluations (t)      : {len(t_eval)}\n"
        f"  Eval Results         : {eval_result}\n"
        f"{'=' * SEP_LINE_LEN}"
    )

    # TODO make dataset specific visualization


    filename = (f"inference_"
                f"{dataset_name}_"
                f"data_params_{data_params_flat}_"
                f"timestamp_{TIMESTAMP}_"
                f"trajectory_model_{traj_model}_"
                f"ode_solver_method_{ode_solver_method}_"
                f"num_reflows_{num_reflows}.png")
    figure_path = os.path.join(INFERENCE_FIGURES_DIR, filename)
    if dataset_name == "skewed_gaussian_2d":
        # TODO add model arch and trajectory modeling approach (grid , continous/space-time)
        Visualizer.plot_with_subplots_gaussian_contour_2d(x0=x0, x1=x1, x1_hat=x1_hat,
                                                          title="Skewed Gaussian Distribution: Target vs. Inferred Samples",
                                                          save_path=figure_path)
    elif dataset_name == "gaussian_mixture_2d":
        Visualizer.plot_with_subplots_gaussian_mixture_2d(x0=x0, x1=x1, x1_hat=x1_hat,
                                                          title="Gaussian Mixture : Target vs. Inferred Samples ",
                                                          save_path=figure_path)
    elif dataset_name in ["circles"]:
        Visualizer.plot_subplots_three_distributions(x0=x0.detach().cpu(), x1=x1.detach().cpu(),
                                                     x1_hat=x1_hat.detach().cpu(),
                                                     save_path=os.path.join(INFERENCE_FIGURES_DIR,
                                                                            f"{dataset_name}_{TIMESTAMP}.png"))
    else:
        raise NotImplementedError("dataset_name has not supported viz")

    logger.info(f"Inference and evaluation process finished successfully!")
