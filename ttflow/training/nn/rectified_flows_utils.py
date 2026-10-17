from typing import Iterable, Tuple
import torch
from loguru import logger
from tqdm import tqdm
from ttflow.data_generation.data_utils import DataUtils
from ttflow.inference.inference_and_evaluation import InferenceHandler
class RectifiedFlowsUtils:
    @staticmethod
    def generate_reflow_data(n: int, dim: int, t_eval: torch.Tensor, vt_model: torch.nn.Module,
                             data_type: torch.dtype, device: torch.device, inference_batch_size: int, ode_solver_method:str) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # get z0,k+1 with fresh noise sampling
        z0 = DataUtils.generate_gaussian_prior(dim=dim, n_samples=n).to(
            dtype=data_type, device=device)
        # get z1 be ODESOLVE
        # batched inference
        num_inference_samples = z0.shape[0]
        inference_handler = InferenceHandler(models=vt_model, inference_mode="one-trajectory")
        z1_batches = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, num_inference_samples, inference_batch_size),
                                  desc="inference-batches"):
                end_idx = min(start_idx + inference_batch_size, num_inference_samples)
                # Slice mini-batch
                z0_batch = z0[start_idx:end_idx, :]
                # Run inference on mini-batch
                traj_batch = inference_handler.infer(x0=z0_batch, t_eval=t_eval, ode_solver_method=ode_solver_method)
                z1_batch = traj_batch[-1].reshape(z0_batch.shape)
                z1_batches.append(z1_batch)
        z1 = torch.cat(z1_batches, dim=0).to(dtype=data_type, device=device)
        return z0, z1
