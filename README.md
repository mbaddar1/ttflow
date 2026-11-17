# TTFLOw

> Lightweight documentation — work in progress.


## Overview

Rectified flows are generative models that learn to transport samples from a source distribution $\pi_0$ to a target distribution $\pi_1$.
Given a pairing of source and target samples $x_0 \sim \pi_0$ and $x_1 \sim \pi_1$, the generation trajectory from $x_0$ to $x_1$ is modeled as an ordinary differential equation (ODE):

$$
\hat{x}_1 = x_0 + \int_{0}^{1} v(x_t, t)\, dt,
$$

where $v(x_t, t)$ is a learned velocity field, $x_t$ is the state at time $t \in [0, 1]$, and $x_0, x_1, \hat{x}_1 \in \mathbb{R}^d$ are $d$-dimensional vectors.
The generated samples $\hat{x}_1$ follow the pushforward distribution $\hat{\pi}_1$, which should approximate $\pi_1$.

**Motivation**

In the original rectified flow paper ([Liu et al., 2022](https://arxiv.org/abs/2209.03003)), the velocity field $v$ is modeled as a neural network, such as a simple feedforward network or a U-Net.
In this project, we explore an alternative: modeling the velocity field with a low-rank **tensor train (TT)** ([Oseledets, 2011](https://doi.org/10.1137/090752286)).

The motivation is twofold:

- **Memory efficiency:** a low-rank TT represents high-dimensional functions with far fewer parameters than a dense representation.
- **Efficient training algorithms:** TTs can be fitted with dedicated solvers such as alternating least squares (ALS) and DMRG, which may offer advantages over gradient-based training of neural networks for the rectified flow objective.

<!-- ✅ done · 🚧 in progress · ⏳ planned -->

## Experiments

| Date | Experiment | Key finding | Details |
|---|---|---|---|
| | | | |

Full details, code links and commit hashes: [experiments.md](experiments.md)

## Developers
1. [mbaddar](https://github.com/mbaddar1)
2. [meigel](https://github.com/meigel)
3. [janinaschutte](https://github.com/janinaschutte)