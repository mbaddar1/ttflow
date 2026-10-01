import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde


class Visualizer:
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
