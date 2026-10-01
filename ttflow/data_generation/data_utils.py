import os.path
from typing import Tuple
import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
from tqdm import tqdm
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3, fetch_california_housing, \
    make_circles, make_swiss_roll, make_moons
from loguru import logger
from ttflow.data_generation.distributions.many_well import ManyWell_Gaussian
from ttflow.data_generation.distributions.multivariate_skewnorm import MultivariateSkewGaussian
from ttflow.viz.viz import Visualizer
from datetime import datetime
FIGURES_DIR = "figures/data_generation"
TIME_STAMP = datetime.now().isoformat()

class DataUtils:
    @staticmethod
    def rectified_flow_get_l2_xy(x0=None, x1=None, time_grid=None):
        """
        This function is to get training input and output for the v(t) model
        See https://arxiv.org/pdf/2209.03003 eq (1)
        Note : it is used for the nn-model training . For the tt-model training , chmda has another piece of code
        flow_matching/learning.py:85
        # TODO unify the input output generation code and test it
        @param z0:
        @param z1:
        @return:
        """
        assert len(x0.shape) == 2
        assert len(x1.shape) == 2
        for i in range(len(x0.shape)):
            assert x0.shape[i] == x1.shape[i]
        if time_grid is None:
            t = torch.rand((x1.shape[0], 1)).type(x0.dtype).to(x0.device)
            x_t = t * x1 + (1. - t) * x0
            target = x1 - x0
            return x_t, t, target
        else:
            # return list of z_t of length t_grid , each of tensor of shape z1.shape (z0.shape)
            x_t_list = []
            for t in tqdm(time_grid, desc="generating xt_grid (over the time-grid)"):
                x_t = t * x1 + (1. - t) * x0
                x_t_list.append(x_t)
            assert len(x_t_list) == len(time_grid)
            target = x1 - x0
            return x_t_list, target

    @staticmethod
    def transform_to_masking_data(x: torch.Tensor, y: torch.Tensor):
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] > 1, "Y dim = 1 , no need for this hassle"
        x_list = []
        y_list = []
        for d in range(y.shape[1]):
            dim_tensor = torch.tensor([d] * x.shape[0]).reshape(-1, 1)
            x_concat_d = torch.concat([dim_tensor, x], dim=1)
            x_list.append(x_concat_d)
            y_list.append(y[:, d])
        x_all = torch.concat(x_list, dim=0)
        y_all = torch.concat(y_list, dim=0).reshape(-1, 1)
        # assert shapes

        assert x_all.shape[1] == x.shape[1] + 1
        assert x_all.shape[0] == y_all.shape[0]
        assert x_all.shape[0] == x.shape[0] * y.shape[1]

        # shuffle
        perm = torch.randperm(x_all.size(0))
        x_shuffled = x_all[perm]
        y_shuffled = y_all[perm]

        assert x_shuffled.shape[0] == y.shape[1] * x.shape[0]
        assert x_shuffled.shape[0] == y_shuffled.shape[0]
        assert y_shuffled.shape[1] == 1
        return x_shuffled, y_shuffled

    @staticmethod
    def get_gaussian_mixture(
            n: int,
            num_components: int = 3,
            radius: float = 10.0,
            variance: float = 0.3,
            rotation: float = torch.pi,
            device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from two 2D Gaussian mixtures arranged on a circle.

        The initial distribution has equally spaced Gaussian components.
        The target distribution is obtained by rotating the component means.

        Args:
            n:
                Number of samples drawn from each mixture.
            num_components:
                Number of Gaussian components.
            radius:
                Distance of each component mean from the origin.
            variance:
                Per-coordinate variance of every Gaussian component.
            rotation:
                Counterclockwise rotation of the target mixture in radians.
            device:
                Device on which samples are generated.
            dtype:
                Floating-point dtype.

        Returns:
            samples_0:
                Samples from the initial mixture, with shape ``(n, 2)``.
            samples_1:
                Samples from the target mixture, with shape ``(n, 2)``.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, but received {n}.")

        if num_components <= 0:
            raise ValueError(
                "num_components must be positive, "
                f"but received {num_components}."
            )

        if radius <= 0:
            raise ValueError(
                f"radius must be positive, but received {radius}."
            )

        if variance <= 0:
            raise ValueError(
                f"variance must be positive, but received {variance}."
            )

        device = torch.device(device)

        # Equally spaced component angles.
        angles_0 = (
                2.0
                * torch.pi
                * torch.arange(num_components, device=device, dtype=dtype)
                / num_components
        )

        angles_1 = angles_0 + rotation

        def means_from_angles(angles: torch.Tensor) -> torch.Tensor:
            return radius * torch.stack(
                (torch.cos(angles), torch.sin(angles)),
                dim=-1,
            )

        means_0 = means_from_angles(angles_0)
        means_1 = means_from_angles(angles_1)

        weights = torch.full(
            (num_components,),
            1.0 / num_components,
            device=device,
            dtype=dtype,
        )

        covariance_matrices = (
                variance
                * torch.eye(2, device=device, dtype=dtype)
                .expand(num_components, 2, 2)
        )

        def create_mixture(means: torch.Tensor) -> MixtureSameFamily:
            mixture_distribution = Categorical(probs=weights)

            component_distribution = MultivariateNormal(
                loc=means,
                covariance_matrix=covariance_matrices,
            )

            return MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        initial_model = create_mixture(means_0)
        target_model = create_mixture(means_1)

        x0 = initial_model.sample((n,))
        x1 = target_model.sample((n,))

        return x0, x1

    @staticmethod
    def rectified_flows_get_inference_x0x1(x_dim: int, y_dim: int, dataset_name: str, n: int, noise: float = 0.01,
                                           random_state: int = 42, data_type=torch.float64, **kwargs):
        if dataset_name == "make_regression":
            x, y = make_regression(
                n_samples=n,
                n_features=x_dim,
                n_targets=y_dim,  # <--- Multiple targets
                noise=noise,
                random_state=random_state,
                effective_rank=4,
                tail_strength=0.5
            )
            non_lin_y = torch.nn.Sigmoid()(torch.tensor(y))
            return torch.tensor(x), torch.tensor(non_lin_y)
        elif dataset_name == "make_friedman1":  # for regression only testing
            raise Exception("Not supported now")
            # x, y = make_friedman1(random_state=random_state, n_features=x_dim, noise=noise, n_samples=n)
            # return torch.tensor(x), torch.tensor(y).reshape(-1, 1)
        elif dataset_name == "make_friedman2":  # for regression only testing
            raise Exception("Not supported now")
            # x, y = make_friedman2(random_state=random_state, noise=noise, n_samples=n)
            # return torch.tensor(x), torch.tensor(y)
        elif dataset_name == "make_friedman3":  # for regression only testing
            raise Exception("Not supported now")
            # x, y = make_friedman3(random_state=random_state, n_samples=n, noise=noise)
            # return torch.tensor(x), torch.tensor(y)
        elif dataset_name == "california_housing":  # for regression only testing
            raise Exception("Not supported now")
            # scaler = StandardScaler()
            # x, y = fetch_california_housing(return_X_y=True)
            # x_scaled = scaler.fit_transform(x)
            # return torch.tensor(x_scaled), torch.tensor(y)
            # splits = dataset_name.split("_")
            # dataset_subname = splits[2]
        elif dataset_name == "circles":
            x1, _ = make_circles(n_samples=n, shuffle=True, noise=noise, random_state=42)
        elif dataset_name == "spiral":
            x1 = make_swiss_roll(n_samples=n, noise=noise)[0][:, [0, 2]]
        elif dataset_name == "moons":
            x1, _ = make_moons(n_samples=n, noise=noise)
        elif dataset_name == "manywell":
            assert "dim" in kwargs.keys() and isinstance(kwargs["dim"], int)
            assert "many_well_dim" in kwargs.keys() and isinstance(kwargs["many_well_dim"], int)
            many_well_gaussian_dist = ManyWell_Gaussian(dim=kwargs["dim"], many_well_dim=kwargs["many_well_dim"],
                                                        normalised=True, use_gpu=False)
            x1 = many_well_gaussian_dist.sample([n])
        elif dataset_name == "gaussian_mixture_2d":
            num_components_key_name = "num_components"
            kwargs_has_ncomps = num_components_key_name in kwargs.keys()
            default_n_comps = 3
            num_components = kwargs[num_components_key_name] if kwargs_has_ncomps else default_n_comps
            if kwargs_has_ncomps:
                logger.info(
                    f"Generating gaussian mixture 2d with num_components = {num_components}")
            else:
                logger.info(f"num_components not provided in kwargs = {kwargs}")
            _, x1 = DataUtils.get_gaussian_mixture(
                n=n,
                num_components=num_components,
                radius=10.0,
                variance=0.3,
                rotation=torch.pi / 3
            )
        elif dataset_name == "skewed_gaussian_2d":
            dist_params = ["mean", "cov", "shape"]
            assert all(
                dist_param in kwargs.keys() for dist_param in dist_params), f"must have params {dist_params}".format(
                dist_params=dist_params)
            mean = kwargs["mean"]
            cov = kwargs["cov"]
            shape = kwargs["shape"]
            mvsg = MultivariateSkewGaussian(mean=mean, cov=cov, shape=shape)
            x1 = mvsg.rvs_fast(size=n)
            Visualizer.plot_density_contours(x=x1, output_filename=os.path.join(FIGURES_DIR, f"skewed_gaussian_2d_{TIME_STAMP}.png"),
                                             title=f"skewed gaussian 2d : mean = {mean},cov = {cov},shape = {shape}")
        # keep the return at the end of the method
        # in this case x0 is multivariate iso Gaussian
        # this is different from the setup in
        # https://colab.research.google.com/drive/1CyUP5xbA3pjH55HDWOA8vRgk2EEyEl_P?usp=sharing
        # where x0 is also a gaussian mixture with different parameters
        else:
            raise NotImplementedError(f"subdataset {dataset_name} not implemented")
        x1 = torch.tensor(x1, dtype=data_type)
        mu_prior = torch.zeros(x1.shape[1])
        cov_prior = torch.eye(x1.shape[1])
        x0 = (torch.distributions.MultivariateNormal(loc=mu_prior,
                                                     covariance_matrix=cov_prior).
              sample(sample_shape=torch.Size((n,)))).type(data_type)
        return x0, x1
    # FIXME if not needed del to avoid confusion
    # @staticmethod
    # def get_rectified_flow_l2_xy(x_dim: int, y_dim: int, dataset_name: str, n: int, noise: float = 0.01, random_state: int = 42,
    #                              time_grid=None, **kwargs):
    #     z0, z1 = DataUtils.get_x0_x1_rectified_flows(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name, n=n,
    #                                                  noise=noise, random_state=random_state, **kwargs)
    #     if time_grid is None:
    #         X, t, y = DataUtils.get_rectified_flow_l2_data_tuple(x0=z0, x1=z1)
    #         X_aug = torch.cat([X, t], dim=1)
    #         return X_aug, y
    #     else:
    #         X, y = DataUtils.get_rectified_flow_l2_data_tuple(x0=z0, x1=z1, time_grid=time_grid)
    #         return X, y
