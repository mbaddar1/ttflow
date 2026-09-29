from typing import Optional, Dict

from seaborn import kdeplot
import torch.distributed

import numpy as np
from tqdm import tqdm
# import fab
import torch.nn as nn
from typing import Callable
# from fab.types_ import LogProbFunc

import torch
from matplotlib import pyplot as plt

"""
The many_well dataset is used in this paper
Sampling from Boltzmann densities with physics informed low-rank formats
https://arxiv.org/pdf/2412.07637
"""
# from fab.target_distributions.base import TargetDistribution
# from fab.utils.training import DatasetIterator
# from fab.target_distributions.double_well import DoubleWellEnergy

def rejection_sampling(n_samples: int, proposal: torch.distributions.Distribution,
                       target_log_prob_fn: Callable, k: float) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample((n_samples * 10,))  # mult by 10 to account for rejection
    u_0 = torch.distributions.Uniform(0, k * torch.exp(proposal.log_prob(z_0))) \
        .sample().to(z_0)
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_log_prob_fn, k)
        samples = torch.concat([samples, new_samples], dim=0)
        return samples


class Energy(torch.nn.Module):
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]


class DoubleWellEnergy(Energy, nn.Module):
    def __init__(self, dim=2, a=-0.5, b=-6.0, c=1.,use_gpu=True):
        assert dim == 2  # We only use the 2D version
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c
        self.use_gpu = use_gpu
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define proposal params
            self.register_buffer("component_mix", torch.tensor([0.2, 0.8]))
            self.register_buffer("means", torch.tensor([-1.7, 1.7]))
            self.register_buffer("scales", torch.tensor([0.5, 0.5]))

    def _energy_dim_1(self, x_1):
        return self._a * x_1 + self._b * x_1.pow(2) + self._c * x_1.pow(4)

    def _energy_dim_2(self, x_2):
        return 0.5 * x_2.pow(2)

    def _energy(self, x):
        x_1 = x[:, 0]
        x_2 = x[:, 1]
        e1 = self._energy_dim_1(x_1)
        e2 = self._energy_dim_2(x_2)
        return e1 + e2

    def log_prob(self, x):
        return torch.squeeze(-self.energy(x))

    def sample_first_dimension(self, shape):
        assert len(shape) == 1
        # see fab.sampling_methods.rejection_sampling_test.py
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define target.
            def target_log_prob(x):
                # log_prob(x1) \probto -U(x1) , see fn _energy1
                return -x ** 4 + 6 * x ** 2 + 1 / 2 * x

            TARGET_Z = 11784.50927

            # Define proposal
            device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
            mix = torch.distributions.Categorical(self.component_mix.to(device))
            com = torch.distributions.Normal(self.means.to(device), self.scales.to(device))

            proposal = torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                             component_distribution=com,validate_args=True)

            k = TARGET_Z * 3

            samples = rejection_sampling(shape[0], proposal, target_log_prob, k)
            return samples
        else:
            raise NotImplementedError

    def sample(self, shape):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            dim1_samples = self.sample_first_dimension(shape)
            dim2_samples = torch.distributions.Normal(
                torch.tensor(0.0).to(dim1_samples.device),
                torch.tensor(1.0).to(dim1_samples.device)
            ).sample(shape)
            return torch.stack([dim1_samples, dim2_samples], dim=-1)
        else:
            raise NotImplementedError

    @property
    def log_Z_2D(self):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            log_Z_dim0 = np.log(11784.50927)
            log_Z_dim1 = 0.5 * np.log(2 * torch.pi)
            return log_Z_dim0 + log_Z_dim1
        else:
            raise NotImplementedError


class ManyWellEnergy(DoubleWellEnergy):  # , TargetDistribution):
    """Many Well target distribution create by repeating the Double Well Boltzmann distribution."""

    def __init__(self, dim=4, use_gpu: bool = True,
                 normalised: bool = False,
                 a=-0.5, b=-6.0, c=1.):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        super(ManyWellEnergy, self).__init__(dim=2, a=a, b=b, c=c,use_gpu=use_gpu)
        self.dim = dim
        self.centre = 1.7
        self.max_dim_for_all_modes = 40  # otherwise we get memory issues on huuuuge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = torch.meshgrid([torch.tensor([-self.centre, self.centre]) for _ in
                                              range(self.n_wells)])
            dim_1_vals = torch.stack([torch.flatten(dim) for dim in dim_1_vals_grid], dim=-1)
            n_modes = 2 ** self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = torch.zeros((n_modes, dim))
            test_set[:, torch.arange(dim) % 2 == 0] = dim_1_vals
            self.register_buffer("_test_set_modes", test_set)
        else:
            print("using test set containing not all modes to prevent memory issues")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]

        if use_gpu:
            if torch.cuda.is_available():
                self.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self.normalised = normalised

    @property
    def log_Z(self):
        return torch.tensor(self.log_Z_2D * self.n_wells)

    @property
    def Z(self):
        return torch.exp(self.log_Z)

    def sample(self, shape):
        """Sample by sampling each pair of dimensions from the double well problem
        using rejection sampling for the first dimension, and exact sampling for the second. """
        return torch.concat([super(ManyWellEnergy, self).sample(shape)
                             for _ in range(self.n_wells)],
                            dim=-1)

    # def get_modes_test_set_iterator(self, batch_size: int):
    #     """Test set created from points manually placed near each mode."""
    #     if self.dim < self.max_dim_for_all_modes:
    #         test_set = self._test_set_modes
    #     else:
    #         outer_batch_size = int(1e4)
    #         test_set = torch.zeros((outer_batch_size, self.dim))
    #         test_set[:, torch.arange(self.dim) % 2 == 0] = \
    #             -self.centre + self.centre * 2 * \
    #             torch.randint(high=2, size=(outer_batch_size, int(self.dim/2)))
    #     return DatasetIterator(batch_size=batch_size, dataset=test_set,
    #                            device=self.device)

    def log_prob(self, x):
        log_prob = torch.sum(
            torch.stack(
                [super(ManyWellEnergy, self).log_prob(x[:, i * 2:i * 2 + 2])
                 for i in range(self.n_wells)]),
            dim=0)
        if self.normalised:
            return log_prob - self.log_Z
        else:
            return log_prob

    # def log_prob_2D(self, x):
    #     # for plotting, given 2D x
    #     return super(ManyWellEnergy, self).log_prob(x)

    # def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
    #                         log_q_fn: Optional[LogProbFunc] = None,
    #                         batch_size: Optional[int] = None) -> Dict:

    #     del samples
    #     n_runs = 50
    #     n_vals_per_split = log_w.shape[0] // n_runs
    #     log_w = log_w[:n_vals_per_split*n_runs]
    #     log_w = torch.stack(log_w.split(n_runs), dim=-1)

    #     # Check accuracy in estimating normalisation constant.
    #     log_Z_estimate = torch.logsumexp(log_w, dim=-1) - np.log(log_w.shape[-1])
    #     relative_error = torch.exp(log_Z_estimate - self.log_Z) - 1
    #     MSE_Z_estimate = torch.mean(torch.abs(relative_error))

    #     abs_MSE_log_Z_estimate = torch.mean(torch.abs(log_Z_estimate - self.log_Z))

    #     info = {}
    #     info.update(relative_MSE_Z_estimate=MSE_Z_estimate.cpu().item())
    #     info.update(abs_MSE_log_Z_estimate=abs_MSE_log_Z_estimate.cpu().item())

    #     if log_q_fn is not None:
    #         # Used later for estimation of test set probabilities.
    #         assert batch_size is not None
    #         n_batches = max(log_w.shape[0] // batch_size, 1)

    #         sum_log_prob = 0.0
    #         sum_log_prob_exact = 0.0
    #         sum_kl_exact = 0.0
    #         test_set_iterator_modes = self.get_modes_test_set_iterator(batch_size=batch_size)

    #         for x in test_set_iterator_modes:
    #             # Mode test set.
    #             log_q_x_modes = torch.sum(log_q_fn(x)).detach().cpu()
    #             sum_log_prob += log_q_x_modes

    #         for _ in range(n_batches):
    #             # Samples from p test set.
    #             x_exact = self.sample((batch_size,))
    #             log_q_x_exact = log_q_fn(x_exact)
    #             sum_log_prob_exact += torch.sum(log_q_x_exact).detach().cpu()
    #             sum_kl_exact += torch.sum(self.log_prob(x_exact) - self.log_Z - log_q_x_exact).detach().cpu()

    #         eval_batch_size = batch_size * n_batches

    #         info.update(
    #             test_set_modes_mean_log_prob=(sum_log_prob / test_set_iterator_modes.test_set_n_points).cpu().item(),
    #             test_set_exact_mean_log_prob=(sum_log_prob_exact / eval_batch_size).cpu().item(),
    #             forward_kl=(sum_kl_exact / eval_batch_size).cpu().item(),
    #             eval_batch_size=eval_batch_size
    #         )
    #     return info


class ManyWell_Gaussian():
    """Many Well target distribution create by repeating the Double Well Boltzmann distribution."""

    def __init__(self, dim=8, use_gpu: bool = True,
                 normalised: bool = False,
                 a=-0.5, b=-6.0, c=1.,
                 many_well_dim=4):
        self.ManyWellDist = ManyWellEnergy(dim=many_well_dim, use_gpu=use_gpu,
                                           normalised=normalised,
                                           a=a, b=b, c=c)
        self.n_wells = self.ManyWellDist.n_wells
        self.normalised = self.ManyWellDist.normalised
        self.many_well_dim = many_well_dim
        self.concat_gaussian = dim > many_well_dim
        if self.concat_gaussian:
            gaussian_device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            gaussian_mean = torch.zeros(size=[dim - many_well_dim], device=gaussian_device)
            gaussian_cov = torch.eye(n=dim - many_well_dim, device=gaussian_device)
            self.GaussianDist = torch.distributions.multivariate_normal.MultivariateNormal(gaussian_mean, gaussian_cov)

    @property
    def log_Z(self):
        return torch.tensor(self.log_Z_2D * self.n_wells)

    def sample(self, shape):
        """Sample by sampling each pair of dimensions from the double well problem
        using rejection sampling for the first dimension, and exact sampling for the second. """
        if self.concat_gaussian:
            return torch.concat([self.ManyWellDist.sample(shape), self.GaussianDist.sample(shape)],dim=-1)
        else:
            return self.ManyWellDist.sample(shape)

    def log_prob(self, x):
        assert len(x.shape) == 2
        log_prob = self.ManyWellDist.log_prob(x[:, :self.many_well_dim])
        if self.concat_gaussian:
            log_prob += self.GaussianDist.log_prob(x[:, self.many_well_dim:])
        return log_prob

    @staticmethod
    def pairwise_plot(sample: torch.Tensor,output_file_name:str):
        assert sample.dim() == 2
        d = sample.shape[1]
        assert d <=8
        assert d%2==0
        num_pairs = d // 2
        X = sample.detach().cpu()
        fig, axes = plt.subplots(1, num_pairs, figsize=(5 * num_pairs, 4))

        if num_pairs == 1:
            axes = [axes]

        for i in range(num_pairs):
            ax = axes[i]
            dim1 = 2 * i
            dim2 = 2 * i + 1

            kdeplot(
                x=X[:, dim1].numpy(),
                y=X[:, dim2].numpy(),
                fill=True,
                cmap="viridis",
                thresh=0.05,
                levels=100,
                ax=ax
            )
            ax.set_title(f"KDE of dims ({dim1},{dim2})")
            ax.set_xlabel(f"x{dim1}")
            ax.set_ylabel(f"x{dim2}")

        plt.tight_layout()
        plt.savefig(output_file_name)



class ManyWell_Gaussian_Jannis():
    """Many Well target distribution create by repeating the Double Well Boltzmann distribution."""

    def __init__(self, dim=16,
                 a=-0.5, b=-6.0, c=1.,
                 many_well_dim=4,use_gpu=False):
        self.OneDimDoubleWell = DoubleWellEnergy(dim=2,
                                                 a=a, b=b, c=c)

        self.many_well_dim = many_well_dim
        self.concat_gaussian = dim > many_well_dim
        if self.concat_gaussian:
            self.GaussianDist = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(dim - many_well_dim), torch.eye(dim - many_well_dim))

    def sample(self, shape):
        """Sample by sampling each pair of dimensions from the double well problem
        using rejection sampling for the first dimension, and exact sampling for the second. """
        samples = torch.stack([self.OneDimDoubleWell.sample_first_dimension(shape) for _ in range(self.many_well_dim)],
                              dim=-1)
        if self.concat_gaussian:
            print(self.GaussianDist.sample(shape).shape)
            samples = torch.concat([samples, self.GaussianDist.sample(shape)], dim=-1)
        return samples

    def log_prob(self, x):
        assert len(x.shape) == 2
        log_prob = torch.sum(
            torch.stack([self.OneDimDoubleWell._energy_dim_1(x[:, i]) for i in range(self.many_well_dim)], dim=-1),
            dim=-1)
        if self.concat_gaussian:
            log_prob += self.GaussianDist.log_prob(x[:, self.many_well_dim:])
        return log_prob
