import os

import torch

from ttflow.viz.viz import Visualizer

if __name__=="__main__":
    # from your_module import Visualizer

    torch.manual_seed(0)
    N, D = 2000, 8

    # Base: standard normal
    x0 = torch.randn(N, D)

    # Target: shifted, scaled, and correlated Gaussian
    A = torch.randn(D, D) * 0.3 + torch.eye(D)  # random mixing matrix
    mu = torch.linspace(-2, 2, D)  # different mean per dimension
    x1 = torch.randn(N, D) @ A.T + mu

    # Inferred: target plus a bit of noise and a small bias (mimics an imperfect model)
    x1_hat = x1 + 0.2 * torch.randn(N, D) + 0.1
    Visualizer.plot_three_samples_nd_with_contours(
        x0=x0.detach().cpu(),
        x1=x1.detach().cpu(),
        x1_hat=x1_hat.detach().cpu(),
        title=f"Multivariate Gaussian (D={D})",
        save_path="test.png",
        # optional overrides:
        # x1_range=(-6, 6), x2_range=(-6, 6),
        # cell_size=1.2, grid_size=60, max_kde_samples=1500,   # faster / smaller
    )
    pass