from typing import Iterable, Tuple
import torch
from loguru import logger
from tqdm import tqdm
from ttflow.data_generation.data_utils import DataUtils
from ttflow.inference.inference_and_evaluation import InferenceHandler
class RectifiedFlowsUtils:
    @staticmethod
    def generate_reflow_data(n, dim, t_eval, vt_model, data_type, device,
                             inference_batch_size, ode_solver_method):
        z0 = DataUtils.generate_gaussian_prior(dim=dim, n_samples=n).to(dtype=data_type, device=device)
        inference_handler = InferenceHandler(models=vt_model, inference_mode="one-trajectory")
        vt_model.eval()
        z1_batches = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, z0.shape[0], inference_batch_size), desc="inference-batches"):
                z0_batch = z0[start_idx:start_idx + inference_batch_size]
                traj_batch = inference_handler.infer(x0=z0_batch, t_eval=t_eval,
                                                     ode_solver_method=ode_solver_method)
                # .cpu() makes a real copy of just the endpoint; .clone() would do the same on GPU
                z1_batches.append(traj_batch[-1].reshape(z0_batch.shape).cpu())
                del traj_batch
        z1 = torch.cat(z1_batches, dim=0).to(dtype=data_type, device=device)
        return z0, z1