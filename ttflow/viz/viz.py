import math
import os
import random
from pathlib import Path
from typing import Optional, Union, List, Sequence
from loguru import logger
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde


class Visualizer:
    @staticmethod
    def plot_subplots_three_distributions(
            x0: torch.Tensor,
            x1: torch.Tensor,
            x1_hat: torch.Tensor,
            max_points: int = 2000,
            color: str = "#1f77b4",
            titles: tuple[str, str, str] = (
                    "Base ($x_0$)",
                    "Target ($x_1$)",
                    "Inferred ($\hat{x}_1$)",
            ),
            save_path: Optional[Union[str, Path]] = None,
            dpi: int = 300,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        # Move tensors to CPU
        tensors = [x0.detach().cpu(), x1.detach().cpu(), x1_hat.detach().cpu()]

        # 1. Exact min and max bounds directly from x1 (no padding, no symmetric padding)
        x1_cpu = tensors[1]
        x_min, y_min = x1_cpu.min(dim=0).values.numpy()
        x_max, y_max = x1_cpu.max(dim=0).values.numpy()

        # 2. Render scatter subplots
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 3.8), sharex=True, sharey=True, dpi=dpi
        )

        for ax, data_tensor, title in zip(axes, tensors, titles):
            if len(data_tensor) > max_points:
                indices = torch.randperm(len(data_tensor))[:max_points]
                pts = data_tensor[indices].numpy()
            else:
                pts = data_tensor.numpy()

            ax.scatter(
                pts[:, 0], pts[:, 1], c=color, s=8, alpha=0.5, edgecolors="none"
            )

            ax.set_title(title, fontsize=12, pad=6)

            # Explicitly clip to x1 bounds to force edge-to-edge rendering
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Remove ticks, margins, and borders
            ax.margins(0)
            ax.set_xticks([])
            ax.set_yticks([])

        # Reduce padding around and between subplots
        plt.tight_layout(pad=0.5)

        # 3. Save plot if path is provided
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)

        return fig, list(axes)

    @staticmethod
    def plot_one_2d_distribution_scatter(
        x: torch.Tensor,
        labels: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        color: str = "#377eb8",
        cmap: str = "tab10",
        s: int = 12,
        alpha: float = 0.8,
        title: str = "2D Scatter Distribution",
        xlabel: Optional[str] = "Dimension 0",
        ylabel: Optional[str] = "Dimension 1",
        grid: bool = True,
        xscale: str = "linear",
        yscale: str = "linear",
        hide_ticks: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plots a 2D scatter plot directly from a PyTorch tensor with full

        customization for scaling, gridlines, labels, and color mapping.
        ref link : https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
        Args:
            x: Tensor of shape (N, 2)
            labels: Optional integer cluster/class labels for multi-color
              plotting
            color: Default color if single-color plot (e.g., '#377eb8')
            cmap: Matplotlib colormap name used when `labels` is provided
            s: Marker size
            alpha: Marker transparency (0.0 to 1.0)
            title: Plot title
            xlabel: Label for X axis (set None to remove)
            ylabel: Label for Y axis (set None to remove)
            grid: Enable subtle background grid
            xscale: Axis scale type ('linear', 'log', 'symlog', 'logit')
            yscale: Axis scale type ('linear', 'log', 'symlog', 'logit')
            hide_ticks: If True, removes axis tick numbers (scikit-learn style)
            save_path: Optional path to save output image
            dpi: Image resolution
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                f"Expected tensor of shape (N, 2), but got {tuple(x.shape)}"
            )

        # Move PyTorch tensor to CPU NumPy
        pts = x.detach().cpu().numpy()

        plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

        # 1. Multi-class vs Single-color plotting
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            scatter = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=labels,
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolors="none",
            )
        else:
            scatter = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                color=color,
                s=s,
                alpha=alpha,
                edgecolors="none",
            )

        # 2. Axis Scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # 3. Grid Settings
        if grid:
            ax.grid(True, linestyle="--", alpha=0.3, color="gray", zorder=0)
            ax.set_axisbelow(True)  # Grid behind points

        # 4. Labels & Titles
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)

        # 5. Tick Visibility
        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        # 6. Save image if requested
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")

        return fig, ax
    @staticmethod
    def plot_with_subplots_gaussian_mixture_2d(
            x0: torch.Tensor,
            x1: torch.Tensor,
            x1_hat: torch.Tensor,
            title: str = "Gaussian Mixture Flow Distributions",
            save_path: str = None,
            bins: int = 120,  # Slightly higher resolution for sharp GMM modes
            levels: int = 12,  # More levels to highlight multiple density peaks
            wspace: float = 0.08
    ):
        """
        Fast 2D contour plotting tailored for Gaussian Mixture models with multiple modes.

        Parameters:
            x0 (torch.Tensor): Base samples, shape (N, 2)
            x1 (torch.Tensor): Target samples, shape (N, 2)
            x1_hat (torch.Tensor): Inferred samples, shape (N, 2)
            title (str): Main title for the figure
            save_path (str, optional): File path to save the generated image
            bins (int): Resolution of the density grid (default 120)
            levels (int): Number of contour lines/levels
            wspace (float): Horizontal spacing between subplots
        """
        x0_cpu = x0.detach().cpu()
        x1_cpu = x1.detach().cpu()
        x1_hat_cpu = x1_hat.detach().cpu()

        # 1. Bounding box across all mixture components
        all_samples = torch.cat([x0_cpu, x1_cpu, x1_hat_cpu], dim=0)
        x_min, x_max = all_samples[:, 0].min().item(), all_samples[:, 0].max().item()
        y_min, y_max = all_samples[:, 1].min().item(), all_samples[:, 1].max().item()

        # Add a small padding around the grid boundaries
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        x_edges = torch.linspace(x_min - pad_x, x_max + pad_x, bins + 1)
        y_edges = torch.linspace(y_min - pad_y, y_max + pad_y, bins + 1)

        # Grid centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        X, Y = torch.meshgrid(x_centers, y_centers, indexing="ij")

        # Fast 2D histogram density estimation
        def get_density(data: torch.Tensor):
            H = torch.histogramdd(data, bins=[x_edges, y_edges]).hist
            return H.numpy()

        H0 = get_density(x0_cpu)
        H1 = get_density(x1_cpu)
        H1_hat = get_density(x1_hat_cpu)

        # 2. Subplots Setup
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)

        datasets = [
            (H0, r"Base Distribution ($x_0$)", "Blues"),
            (H1, r"Target GMM ($x_1$)", "Oranges"),
            (H1_hat, r"Inferred GMM ($\hat{x}_1$)", "viridis"),
        ]

        for ax, (H, sub_title, cmap) in zip(axes, datasets):
            ax.contourf(X.numpy(), Y.numpy(), H, levels=levels, cmap=cmap)
            ax.contour(X.numpy(), Y.numpy(), H, levels=levels, colors="black", linewidths=0.3, alpha=0.5)
            ax.set_title(sub_title, fontsize=14, pad=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("$dim_1$", fontsize=12)

        axes[0].set_ylabel("$dim_2$", fontsize=12)
        fig.suptitle(title, fontsize=16, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved successfully to {save_path}")

        plt.show()

    @staticmethod
    def plot_with_subplots_gaussian_contour_2d(
            x0: torch.Tensor,
            x1: torch.Tensor,
            x1_hat: torch.Tensor,
            title: str = "Flow Matching Distributions",
            save_path: str = None,
            bins: int = 100,
            levels: int = 10,
            wspace: float = 0.01  # Control spacing between subplots here (smaller = closer)
    ):
        """
        Fast 2D contour plotting for large sample sizes accepting PyTorch tensors.
        """
        x0_cpu = x0.detach().cpu()
        x1_cpu = x1.detach().cpu()
        x1_hat_cpu = x1_hat.detach().cpu()

        # 1. Compute global bounds across all distributions
        all_samples = torch.cat([x0_cpu, x1_cpu, x1_hat_cpu], dim=0)
        x_min, x_max = all_samples[:, 0].min().item(), all_samples[:, 0].max().item()
        y_min, y_max = all_samples[:, 1].min().item(), all_samples[:, 1].max().item()

        # Bin edges
        x_edges = torch.linspace(x_min, x_max, bins + 1)
        y_edges = torch.linspace(y_min, y_max, bins + 1)

        # Grid centers for contour meshgrid
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        X, Y = torch.meshgrid(x_centers, y_centers, indexing="ij")

        # Helper function for fast 2D histogram density estimation
        def get_density(data: torch.Tensor):
            H = torch.histogramdd(data, bins=[x_edges, y_edges]).hist
            return H.numpy()

        # Compute 2D histograms
        H0 = get_density(x0_cpu)
        H1 = get_density(x1_cpu)
        H1_hat = get_density(x1_hat_cpu)

        # 2. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)

        datasets = [
            (H0, r"Base Distribution ($x_0$)", "Blues"),
            (H1, r"Target Distribution ($x_1$)", "Oranges"),
            (H1_hat, r"Inferred Distribution ($\hat{x}_1$)", "viridis"),
        ]

        for ax, (H, sub_title, cmap) in zip(axes, datasets):
            ax.contourf(X.numpy(), Y.numpy(), H, levels=levels, cmap=cmap)
            ax.contour(X.numpy(), Y.numpy(), H, levels=levels, colors="black", linewidths=0.3, alpha=0.5)
            ax.set_title(sub_title, fontsize=14, pad=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("$dim_1$", fontsize=12)

        axes[0].set_ylabel("$dim_2$", fontsize=12)
        fig.suptitle(title, fontsize=16, y=0.98)

        # Adjust layout and reduce horizontal gap (wspace)
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved successfully to {save_path}")

        plt.show()

    @staticmethod
    def plot_density_contours(
            x,
            ax=None,
            grid_size=300,
            n_levels=5,
            title="2D Gaussian Density Distribution",
            output_filename="density_contours.png",
            sigma=9.0,
    ):
        """Renders 2D density matching the reference style:

        - Viridis heatmap on black background
        - Axis numbers/ticks from -3 to 3
        - Outer white to inner dark contours
        - Centered mean marker
        - Formatted x1, x2 labels and multi-line title
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")

        ax.set_facecolor("black")

        # 1. Coordinate extent [-3, 3] matching the reference
        extent = [-3, 3, -3, 3]

        # 2. Fast 2D histogram binning + Gaussian smoothing (handles large datasets)
        H, xedges, yedges = np.histogram2d(
            x[:, 0],
            x[:, 1],
            bins=grid_size,
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
        )

        Zi = gaussian_filter(H.T, sigma=sigma)

        # 3. Viridis heatmap
        ax.imshow(
            Zi,
            extent=extent,
            origin="lower",
            cmap="viridis",
            aspect="equal",
            interpolation="bicubic",
        )

        # 4. Contours: Outer white (1.0) -> inner black/dark (0.05)
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        Xi, Yi = np.meshgrid(xc, yc)

        levels = np.linspace(Zi.max() * 0.10, Zi.max() * 0.90, n_levels)
        shades = np.linspace(1.0, 0.05, n_levels)
        contour_colors = [f"{s:.2f}" for s in shades]

        ax.contour(
            Xi, Yi, Zi, levels=levels, colors=contour_colors, linewidths=1.3
        )

        # 5. Center dot marker at the empirical mean
        mean_point = x.mean(axis=0)
        ax.scatter(
            mean_point[0],
            mean_point[1],
            color="#333333",
            edgecolors="black",
            s=40,
            zorder=5,
        )

        # 6. Ticks and limits
        ticks = np.arange(-3, 4, 1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # 7. Labels & Title
        ax.set_xlabel(r"$x_1$", fontsize=16, labelpad=5)
        ax.set_ylabel(r"$x_2$", fontsize=16, labelpad=5)
        ax.tick_params(axis="both", which="major", labelsize=14, direction="out")

        if title:
            ax.set_title(title, fontsize=14, pad=10)

        # 8. Border styling
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

        if standalone:
            plt.tight_layout()
            plt.savefig(
                output_filename,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            plt.show()

    @staticmethod
    def plot_two_2d_distributions(
            x0: torch.Tensor,
            x1: torch.Tensor,
            label0: str = "x_0",
            label1: str = "x_1",
            title: str = "Distribution comparison",
            output_file_name: str = "gaussian_mixture.png",
    ) -> None:
        """Plot two 2D distributions with a luminous beam/glow aesthetic.

        :param title:
        :type title:
        :param x0: Tensor of shape (N, 2)
        :param x1: Tensor of shape (M, 2)
        :param label0: LaTeX/string identifier for x0
        :param label1: LaTeX/string identifier for x1
        :param output_file_name: File path to save output image
        """
        # Detach and move to CPU numpy
        pts0 = x0.detach().cpu().numpy()
        pts1 = x1.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0a0b10")
        ax.set_facecolor("#0a0b10")

        # Beam palette: Neon Cyan (x0) and Neon Amber / Magenta (x1)
        c0_glow, c0_core = "#00d4ff", "#e0f7ff"
        c1_glow, c1_core = "#ff3366", "#ffe0eb"

        # --- Beam 0: Layered glow + core ---
        ax.scatter(
            pts0[:, 0],
            pts0[:, 1],
            s=45,
            color=c0_glow,
            alpha=0.08,
            edgecolors="none",
        )
        ax.scatter(
            pts0[:, 0],
            pts0[:, 1],
            s=12,
            color=c0_glow,
            alpha=0.35,
            edgecolors="none",
        )
        ax.scatter(
            pts0[:, 0],
            pts0[:, 1],
            s=2,
            color=c0_core,
            alpha=0.85,
            edgecolors="none",
            label=rf"${label0}$",
        )

        # --- Beam 1: Layered glow + core ---
        ax.scatter(
            pts1[:, 0],
            pts1[:, 1],
            s=45,
            color=c1_glow,
            alpha=0.08,
            edgecolors="none",
        )
        ax.scatter(
            pts1[:, 0],
            pts1[:, 1],
            s=12,
            color=c1_glow,
            alpha=0.35,
            edgecolors="none",
        )
        ax.scatter(
            pts1[:, 0],
            pts1[:, 1],
            s=2,
            color=c1_core,
            alpha=0.85,
            edgecolors="none",
            label=rf"${label1}$",
        )

        # Subtle dark-mode gridlines
        ax.grid(True, linestyle="--", linewidth=0.5, color="#222638", alpha=0.7)

        # Axes & labels styling
        ax.tick_params(colors="#8892b0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#222638")
            spine.set_linewidth(1.0)

        ax.set_xlabel(r"$x_1$", color="#ccd6f6", fontsize=12, labelpad=8)
        ax.set_ylabel(r"$x_2$", color="#ccd6f6", fontsize=12, labelpad=8)
        ax.set_title(title,
                     color="#e6f1ff",
                     fontsize=14,
                     pad=14,
                     fontweight="medium",
                     )

        # Styled legend
        legend = ax.legend(
            frameon=True,
            facecolor="#121524",
            edgecolor="#2a304a",
            fontsize=11,
            markerscale=5.0,  # Ensure legend markers are visible despite tiny scatter size
            loc="upper right",
        )
        for text in legend.get_texts():
            text.set_color("#ccd6f6")

        ax.axis("equal")
        plt.tight_layout()

        plt.savefig(
            output_file_name,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.show()
        plt.close(fig)

    @staticmethod
    def plot_three_2d_distributions(
            title: str,
            x0: torch.Tensor,
            x1: torch.Tensor,
            x2: torch.Tensor,
            label0: str = "x0",
            label1: str = "x1",
            label2: str = "x2",
            output_file_name: str = "gaussian_mixture_3way.png",
    ) -> None:
        """Plot three 2D distributions with a luminous beam/glow aesthetic.

        :param title:
        :type title:
        :param x0: Tensor of shape (N, 2)
        :param x1: Tensor of shape (M, 2)
        :param x2: Tensor of shape (K, 2)
        :param label0: LaTeX/string identifier for x0
        :param label1: LaTeX/string identifier for x1
        :param label2: LaTeX/string identifier for x2
        :param output_file_name: File path to save output image
        """
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0a0b10")
        ax.set_facecolor("#0a0b10")

        # Beam palettes: Neon Cyan, Neon Magenta/Pink, Neon Amber/Gold
        distributions = [
            {
                "pts": x0.detach().cpu().numpy(),
                "glow": "#00d4ff",
                "core": "#e0f7ff",
                "label": label0,
            },
            {
                "pts": x1.detach().cpu().numpy(),
                "glow": "#ff3366",
                "core": "#ffe0eb",
                "label": label1,
            },
            {
                "pts": x2.detach().cpu().numpy(),
                "glow": "#ffb703",
                "core": "#fff6db",
                "label": label2,
            },
        ]

        # Render layered glow + core for each distribution
        for dist in distributions:
            pts = dist["pts"]
            glow_color = dist["glow"]
            core_color = dist["core"]

            # Outer diffuse aura
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=45,
                color=glow_color,
                alpha=0.08,
                edgecolors="none",
            )
            # Mid-layer halo
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=12,
                color=glow_color,
                alpha=0.35,
                edgecolors="none",
            )
            # High-intensity pinpoint core
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=2,
                color=core_color,
                alpha=0.85,
                edgecolors="none",
                label=rf"${dist['label']}$",
            )

        # Subtle dark-mode gridlines
        ax.grid(True, linestyle="--", linewidth=0.5, color="#222638", alpha=0.7)

        # Axes & labels styling
        ax.tick_params(colors="#8892b0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#222638")
            spine.set_linewidth(1.0)

        ax.set_xlabel(r"$x_1$", color="#ccd6f6", fontsize=12, labelpad=8)
        ax.set_ylabel(r"$x_2$", color="#ccd6f6", fontsize=12, labelpad=8)
        ax.set_title(
            title,
            color="#e6f1ff",
            fontsize=14,
            pad=14,
            fontweight="medium",
        )

        # Styled legend
        legend = ax.legend(
            frameon=True,
            facecolor="#121524",
            edgecolor="#2a304a",
            fontsize=11,
            markerscale=5.0,
            loc="upper right",
        )
        for text in legend.get_texts():
            text.set_color("#ccd6f6")

        ax.axis("equal")
        plt.tight_layout()

        plt.savefig(
            output_file_name,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.show()
        plt.close(fig)

    @staticmethod
    def plot_trajectory_1d_vs_time(
            trajectory: List[torch.Tensor],
            t_evals: Union[List[float], torch.Tensor],
            output_path: str = "rectified_flow_trajectories.png",
    ) -> None:
        """Plots and saves 1D trajectory curves x_t^{(0)} and x_t^{(1)} over time t.

        Args:
            trajectory: List of length T containing Tensors of shape (N, 2).
            t_evals: List or Tensor of shape (T,) containing time evaluation points.
            output_path: File path where the plot will be saved.
        """
        # Stack list into tensor of shape (T, N, 2)
        assert len(trajectory) == len(t_evals)
        traj_tensor = (
            torch.stack(trajectory, dim=0)
            if isinstance(trajectory, list)
            else trajectory
        )

        T, N, num_dims = traj_tensor.shape
        assert num_dims == 2, f"Expected 2D tensors (N, 2), but got {num_dims}D."

        # Convert tensors to numpy arrays for matplotlib
        traj_np = traj_tensor.detach().cpu().numpy()
        if isinstance(t_evals, torch.Tensor):
            t_evals = t_evals.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        dim_titles = ["Dimension 0 ($j=0$)", "Dimension 1 ($j=1$)"]

        colors = plt.cm.viridis([i / max(1, N - 1) for i in range(N)])

        for j in range(2):
            ax = axes[j]
            for idx in range(N):
                x_j = traj_np[:, idx, j]
                color = colors[idx]

                # Trajectory line across t
                ax.plot(t_evals, x_j, color=color, alpha=0.6, linewidth=1.2)

                # Leftmost x0(j) at t=0 and Rightmost x1(j) at t=1
                ax.scatter(t_evals[0], x_j[0], color=color, s=20, zorder=3)
                ax.scatter(t_evals[-1], x_j[-1], color=color, s=20, zorder=3)

            ax.set_title(dim_titles[j], fontsize=12)
            ax.set_xlabel("Time ($t$)", fontsize=10)
            ax.set_ylabel(f"$x_t^{{({j})}}$", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(
            f"Rectified Flow Trajectories ($N={N}$ Samples)", fontsize=13, y=1.01
        )
        plt.tight_layout()

        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved trajectory plot to: {output_path}")

    @staticmethod
    def sample_trajectory(
            trajectory: List[torch.Tensor],
            sample_size: int = 30,
            seed: Union[int, None] = None,
    ) -> List[torch.Tensor]:
        """Randomly samples a subset of indices across all z_t in a trajectory list.

        Args:
            trajectory: List of length T where each element z_t is a Tensor of shape
              (N, D).
            sample_size: Number of samples to draw from N (default 30).
            seed: Optional random seed for reproducible sampling.

        Returns:
            List of length T where each sliced z_t is of shape (sample_size, D).
        """
        assert len(trajectory) > 0, "Trajectory list is empty."

        N = trajectory[0].shape[0]
        assert sample_size <= N, (
            f"Requested sample_size ({sample_size}) exceeds total samples N ({N})."
        )

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # Generate random indices along the batch dimension
        indices = torch.randperm(N)[:sample_size]

        # Slice each z_t along dimension 0
        sampled_trajectory = [
            zt[indices] if isinstance(zt, torch.Tensor) else zt[indices.numpy()]
            for zt in trajectory
        ]

        return sampled_trajectory

    @staticmethod
    def plot_trajectory_2d_space(
            trajectory: Union[List[torch.Tensor], torch.Tensor],
            output_path: str = "rectified_flow_2d_space.png",
            max_trajectories: Optional[int] = None,
            subplot_size: float = 4.0,
    ) -> None:
        """Plots 2D spatial paths of trajectories from z0 to z1 for up to 16 dims.

        Dimensions are partitioned into consecutive pairs (0,1), (2,3), ... and each
        pair is drawn in its own subplot. The grid layout is chosen automatically.

        Args:
            trajectory: List of length T containing Tensors of shape (N, D), or a
                Tensor of shape (T, N, D), with D even and 2 <= D <= 16.
            output_path: File path where the plot will be saved.
            max_trajectories: Optionally limit the number of trajectories drawn
                (the first K) to keep the plots readable.
            subplot_size: Side length (inches) of each subplot.
        """
        traj_tensor = (
            torch.stack(trajectory, dim=0)
            if isinstance(trajectory, (list, tuple))
            else trajectory
        )

        T, N, D = traj_tensor.shape
        assert D % 2 == 0, f"Number of dimensions must be even, but got D={D}."
        assert 2 <= D <= 16, f"Expected 2 <= D <= 16 dimensions, but got D={D}."

        if max_trajectories is not None:
            N = min(N, max_trajectories)
            traj_tensor = traj_tensor[:, :N]

        traj_np = traj_tensor.detach().cpu().numpy()

        # Partition dimensions into consecutive pairs: (0,1), (2,3), ...
        pairs = [(i, i + 1) for i in range(0, D, 2)]
        num_plots = len(pairs)  # == D // 2

        # Automatic grid: near-square, columns >= rows
        ncols = math.ceil(math.sqrt(num_plots))
        nrows = math.ceil(num_plots / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(subplot_size * ncols, subplot_size * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        colors = plt.cm.viridis([i / max(1, N - 1) for i in range(N)])

        for plot_idx, (d_x, d_y) in enumerate(pairs):
            ax = axes_flat[plot_idx]

            for idx in range(N):
                x_coords = traj_np[:, idx, d_x]
                y_coords = traj_np[:, idx, d_y]
                color = colors[idx]

                # 2D spatial trajectory curve
                ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=1.2)

                # Start point z0 (t=0)
                ax.scatter(
                    x_coords[0],
                    y_coords[0],
                    color=color,
                    marker="o",
                    s=30,
                    zorder=4,
                    edgecolors="k",
                    linewidth=0.5,
                    label="Start ($z_0$)" if idx == 0 else "",
                )

                # End point z1 (t=1)
                ax.scatter(
                    x_coords[-1],
                    y_coords[-1],
                    color=color,
                    marker="X",
                    s=40,
                    zorder=4,
                    edgecolors="k",
                    linewidth=0.5,
                    label="End ($z_1$)" if idx == 0 else "",
                )

            ax.set_title(f"Dims ({d_x}, {d_y})", fontsize=11)
            ax.set_xlabel(f"Dimension {d_x} ($z^{{({d_x})}}$)", fontsize=10)
            ax.set_ylabel(f"Dimension {d_y} ($z^{{({d_y})}}$)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_aspect("equal", adjustable="datalim")
            if plot_idx == 0:
                ax.legend(loc="best", fontsize=8)

        # Hide unused axes (e.g., 3 pairs -> 2x2 grid leaves one empty)
        for ax in axes_flat[num_plots:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Rectified Flow Spatial Trajectories ($N={N}$, $D={D}$)", fontsize=14
        )
        plt.tight_layout()

        # Save output
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {num_plots} 2D spatial trajectory plots to: {output_path}")

    @staticmethod
    def plot_three_gaussian_dist(
            tensors,
            titles,
            big_title,
            save_path=None,
    ):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 4.5),
            constrained_layout=True,
        )

        for ax, tensor, title in zip(axes, tensors, titles):
            Z = (
                tensor.detach().cpu().numpy()
                if hasattr(tensor, "detach")
                else np.asarray(tensor)
            )

            # Filled density
            ax.contourf(Z, levels=100, cmap="viridis")

            # Fine, subtle contour lines
            zmax = np.nanmax(Z)
            if zmax > 0:
                levels = np.geomspace(zmax * 1e-4, zmax * 0.95, 90)
                ax.contour(
                    Z,
                    levels=levels,
                    colors="#b8a0c8",
                    linewidths=0.35,
                    alpha=0.55,
                )

            ax.set_title(title, fontsize=17, pad=8)
            ax.set_facecolor("#440154")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

        fig.suptitle(big_title, fontsize=23)

        # Save the figure if a path is provided
        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

        return fig, axes

    @staticmethod
    def plot_one_distribution_2d_gaussian(
            x: torch.Tensor,
            title: str,
            save_path: str = None,
            grid_size: int = 250,
            levels: int = 45,
    ):
        x = x.detach().cpu().numpy()

        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape (N, 2)")

        margin = 0.15 * np.ptp(x, axis=0)
        margin = np.maximum(margin, 1e-3)

        x0 = np.linspace(
            x[:, 0].min() - margin[0],
            x[:, 0].max() + margin[0],
            grid_size,
        )
        x1 = np.linspace(
            x[:, 1].min() - margin[1],
            x[:, 1].max() + margin[1],
            grid_size,
        )

        X0, X1 = np.meshgrid(x0, x1)

        # Kernel Density Estimation
        kde = gaussian_kde(x.T)
        Z = kde(np.vstack((X0.ravel(), X1.ravel()))).reshape(X0.shape)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_facecolor("#440154")

        ax.contourf(X0, X1, Z, levels=levels, cmap="viridis")
        ax.contour(
            X0, X1, Z,
            levels=levels,
            colors="white",
            linewidths=0.25,
            alpha=0.45,
        )

        ax.set_title(title, fontsize=22)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.tick_params(axis="both", labelsize=11)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()
        return fig, ax

    # ----------------------------------------------------------------------
    # Router method to plot three distributions4
    # This method is to plot x0,x1,x1_hat for 2D and other higher dimensional with pair-wise approach
    # ----------------------------------------------------------------------
    @classmethod
    def plot_three_samples_nd_with_contours(
            cls,
            x0: torch.Tensor,
            x1: torch.Tensor,
            x1_hat: torch.Tensor,
            title: str,
            save_path: Union[str, None] = None,
            x0_range: Optional[tuple] = None,
            x1_range: Optional[tuple] = None,
            x1_hat_range: Optional[tuple] = None,
            **kwargs,
    ):
        """Dispatch on data dimensionality.

        D == 2 -> side-by-side 2D contour plot (_plot_three_strict_2d).
        D > 2  -> three panels, each a grid of contour plots for the
                  disjoint consecutive pairs (0,1), (2,3), (4,5), ...
                  (_plot_three_pairwise).

        x0_range / x1_range / x1_hat_range: (min, max) used for BOTH axes of
        every subplot belonging to that distribution (base / target / inferred).
        If None, the range is computed automatically from the data.
        Extra kwargs (grid_size, levels, margin_ratio, ...) go to the sub-method.
        """
        if any(x.ndim != 2 for x in (x0, x1, x1_hat)):
            raise ValueError("All distributions must have shape (N, D)")

        dims = {x0.shape[-1], x1.shape[-1], x1_hat.shape[-1]}
        if len(dims) != 1:
            raise ValueError(f"All distributions must share the same D, got {dims}")

        d = dims.pop()
        if d < 2:
            raise ValueError("Need at least 2 dimensions to plot")

        plot_fn = cls._plot_three_strict_2d if d == 2 else cls._plot_three_pairwise
        return plot_fn(
            x0, x1, x1_hat, title, save_path,
            x0_range=x0_range, x1_range=x1_range, x1_hat_range=x1_hat_range,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------
    @staticmethod
    def _kde_contour(ax, samples_2d: np.ndarray, X1, X2, levels: int,
                     max_kde_samples: Optional[int] = None):
        if max_kde_samples is not None and len(samples_2d) > max_kde_samples:
            idx = np.random.choice(len(samples_2d), max_kde_samples, replace=False)
            samples_2d = samples_2d[idx]

        grid = np.vstack((X1.ravel(), X2.ravel()))
        Z = gaussian_kde(samples_2d.T)(grid).reshape(X1.shape)

        ax.set_facecolor("#440154")
        ax.contourf(X1, X2, Z, levels=levels, cmap="viridis")
        ax.contour(X1, X2, Z, levels=levels, colors="white",
                   linewidths=0.25, alpha=0.45)

    # ------------------------------------------------------------------
    # Sub-method 1: strictly 2D
    # ------------------------------------------------------------------
    @classmethod
    def _plot_three_strict_2d(
            cls,
            x0, x1, x1_hat,
            title: str,
            save_path: Union[str, None] = None,
            grid_size: int = 250,
            levels: int = 45,
            x0_range: Optional[tuple] = None,
            x1_range: Optional[tuple] = None,
            x1_hat_range: Optional[tuple] = None,
            margin_ratio: float = 0.15,
    ):
        distributions = [
            (x0, "Base Distribution", x0_range),
            (x1, "Target", x1_range),
            (x1_hat, "Inferred", x1_hat_range),
        ]

        # Automatic range (used where a range is None): shared by both axes
        all_samples = torch.cat([x.detach().cpu() for x, _, _ in distributions], dim=0)
        lo, hi = all_samples.min().item(), all_samples.max().item()
        m = max(margin_ratio * (hi - lo), 1e-3)
        auto_range = (lo - m, hi + m)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (x, subtitle, rng) in zip(axes, distributions):
            rng = rng if rng is not None else auto_range

            X1, X2 = np.meshgrid(np.linspace(*rng, grid_size),
                                 np.linspace(*rng, grid_size))
            cls._kde_contour(ax, x.detach().cpu().numpy(), X1, X2, levels)
            ax.set_xlim(rng)
            ax.set_ylim(rng)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(subtitle, fontsize=18)
            ax.set_xlabel(r"$x_1$", fontsize=14)
            ax.set_ylabel(r"$x_2$", fontsize=14)
            ax.tick_params(axis="both", labelsize=11)

        fig.suptitle(title, fontsize=22)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        return fig, axes

    # ------------------------------------------------------------------
    # Sub-method 2: D > 2, 3 outer panels, each a grid of consecutive pairs
    # ------------------------------------------------------------------
    @classmethod
    def _plot_three_pairwise(
            cls,
            x0, x1, x1_hat,
            title: str,
            save_path: Union[str, None] = None,
            grid_size: int = 100,
            levels: int = 30,
            x0_range: Optional[tuple] = None,
            x1_range: Optional[tuple] = None,
            x1_hat_range: Optional[tuple] = None,
            margin_ratio: float = 0.15,
            max_kde_samples: Optional[int] = 3000,
            cell_size: float = 2.6,  # inches per inner subplot
    ):
        distributions = [
            (x0, "Base Distribution", x0_range),
            (x1, "Target", x1_range),
            (x1_hat, "Inferred", x1_hat_range),
        ]
        data = [x.detach().cpu().numpy() for x, _, _ in distributions]
        d = data[0].shape[1]

        # Disjoint consecutive pairs: (0,1), (2,3), (4,5), ...
        pairs = [(k, k + 1) for k in range(0, d - 1, 2)]
        if d % 2 == 1:
            warnings.warn(f"D={d} is odd; last dimension x_{d - 1} is not plotted.")
        n_pairs = len(pairs)

        n_cols = int(np.ceil(np.sqrt(n_pairs)))
        n_rows = int(np.ceil(n_pairs / n_cols))

        # Automatic range per dimension (used where a range is None):
        # computed from target and inferred only, shared by the three panels
        ref = np.concatenate(data[1:], axis=0)

        def _auto_range(k: int) -> tuple:
            lo, hi = ref[:, k].min(), ref[:, k].max()
            m = max(margin_ratio * (hi - lo), 1e-3)
            return (lo - m, hi + m)

        panel_w, panel_h = n_cols * cell_size, n_rows * cell_size
        fig = plt.figure(figsize=(3 * panel_w + 2.0, panel_h + 1.4))

        outer = GridSpec(1, 3, figure=fig, wspace=0.22,
                         left=0.05, right=0.99, bottom=0.10, top=0.84)

        axes_all = []
        for k, ((_, subtitle, rng), samples) in enumerate(zip(distributions, data)):
            inner = GridSpecFromSubplotSpec(
                n_rows, n_cols, subplot_spec=outer[k], wspace=0.40, hspace=0.45
            )

            bbox = outer[k].get_position(fig)
            fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.02, subtitle,
                     ha="center", va="bottom", fontsize=18)

            panel_axes = {}
            for p, (i, j) in enumerate(pairs):
                # Fixed range for this distribution applies to both axes
                h_range = rng if rng is not None else _auto_range(i)
                v_range = rng if rng is not None else _auto_range(j)

                ax = fig.add_subplot(inner[p // n_cols, p % n_cols])

                X1, X2 = np.meshgrid(np.linspace(*h_range, grid_size),
                                     np.linspace(*v_range, grid_size))
                cls._kde_contour(ax, samples[:, [i, j]], X1, X2, levels,
                                 max_kde_samples=max_kde_samples)
                ax.set_xlim(h_range)
                ax.set_ylim(v_range)
                ax.tick_params(axis="both", labelsize=8)
                ax.set_xlabel(rf"$x_{{{i}}}$", fontsize=11)
                ax.set_ylabel(rf"$x_{{{j}}}$", fontsize=11)

                panel_axes[(i, j)] = ax
            axes_all.append(panel_axes)

        fig.suptitle(title, fontsize=22)
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        return fig, axes_all  # axes_all[k][(i, j)] -> Axes