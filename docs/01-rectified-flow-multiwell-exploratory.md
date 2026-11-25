# 01 — Rectified Flow on the ManyWell Problem: An Exploratory Experiment

> **Series:** TTFlow technical notes, article 01
> **Type:** exploratory experiment
> **Status:** draft
> **Guide / companion reading:** [*Flow With What You Know*, ICLR 2025 Blogposts track](https://iclr-blogposts.github.io/2025/blog/flow-with-what-you-know/) ([Colab](https://tinyurl.com/iclr-2025-flow-blogpost-colab))
> **Benchmark source:** Gruhlke, Berner, Sommer, Richter, *Tensor Train Diffusion: Leveraging Low-Rank Structures for High-Dimensional Score-Based Sampling*, ICML 2026, [arXiv:2607.06841](https://arxiv.org/abs/2607.06841), Section 4.1

---

## 1. Why this note exists

TTFlow approximates rectified-flow velocity fields with functional tensor trains (FTT).
Before we put an FTT anywhere, we need a **neural-network rectified flow teacher** that works on the target
densities we want to study. This note covers that first step:

1. What a plain (1-)rectified flow is, explained with the intuitions from the ICLR blog post.
2. The high-dimensional Multiwell benchmark from Section 4.1 of the Tensor Train Diffusion (TTD) paper.
3. How we set the benchmark up in this repo, and how it differs from the paper.
4. Which questions the teacher experiments answer for the later FTT work.

One clarification first. **The TTD paper does not use rectified flow.** It solves a Hamilton–Jacobi–Bellman
(HJB) PDE for the log-density of a time-reversed noising SDE, with a functional tensor train fitted by ALS
and no neural network. We take only its **benchmark targets** and its **low-rank viewpoint**. The transport
model, rectified flow, is our own choice.

---

## 2. Rectified flow in one page (after the blog post)

The blog post explains flow matching with a physics picture: a **velocity field** says which way every
point should move at time $t$, and following it traces **streamlines**. Streamlines of a well-behaved
field never cross, so the map is invertible. The blog post also says that "flow matching" and
"rectified flow" are the same model, and that "rectified" describes what training does, not a separate step.

### 2.1 Training: random pairs, straight lines, regress the direction

Draw $x_0 \sim \pi_0$ (source, here $\mathcal N(0, I_d)$) and $x_1 \sim \pi_1$ (target), **paired at random**.
Connect them with a straight line at constant speed:

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad t \sim \mathcal U[0,1].
$$

The velocity along that line is the constant $x_1 - x_0$. Train a network to predict it:

$$
\mathcal L(\theta) = \mathbb E_{t,\,x_0,\,x_1}\big\| v_\theta(x_t, t) - (x_1 - x_0) \big\|^2 .
$$

The training lines cross each other. The network cannot satisfy every crossing line at the same point, so
it learns their **conditional mean**, $v^\star(x,t) = \mathbb E[x_1 - x_0 \mid x_t = x]$. That averaged
field has non-crossing streamlines. This is the "rectification". The method is **simulation-free**:
training never integrates an ODE.

### 2.2 Sampling: integrate the ODE

$$
\frac{\mathrm d x_t}{\mathrm d t} = v_\theta(x_t, t), \qquad \hat x_1 = x_0 + \int_0^1 v_\theta(x_t,t)\,\mathrm dt .
$$

The blog post starts with forward Euler, then argues that RK4 is more accurate at the same total cost, because
smooth flow trajectories reward higher-order solvers. It also suggests a time warp that concentrates steps
around $t = 1/2$:

$$
f(t) = 4(1-s)t^3 + 6(s-1)t^2 + (3-2s)t, \qquad s \in [0, 3/2],
$$

where $s = 1.5$ approximates the cosine schedule $(1-\cos\pi t)/2$.

### 2.3 Reflow: straighten by re-pairing

1-rectified flow paths are curved. The blog post shows them "moving inward and back out". **Reflow** replaces
the random pairs with pairs $(x_0, \hat x_1)$, where $\hat x_1$ comes from integrating the trained teacher.
A student, initialised as a copy of the teacher, is then trained with the same loss. The reflowed field is
close to time-independent and its paths are close to straight, so a few Euler steps are enough. This links to
dynamic optimal transport (Benamou–Brenier), where optimal paths are constant-speed straight lines.

In this repo:

| Name | Meaning |
|---|---|
| 1-RecFlow | rectified flow, 0 reflows |
| 2-RecFlow | 1 reflow on teacher-generated pairs |

---

## 3. The Multiwell benchmark (TTD paper, Section 4.1)

### 3.1 Target density

TTD Eq. (18):

$$
\rho_{\text{target}}(x) = \exp\Big(-\sum_{i=1}^{m}\big(x_i^2-\delta\big)^2 \;-\; \tfrac12 \sum_{i=m+1}^{d} x_i^2\Big),
\qquad x \in \mathbb R^d .
$$

- The first $m$ coordinates are **symmetric double wells** with minima at $x_i = \pm\sqrt\delta$.
- The other $d-m$ coordinates are **standard Gaussian**.
- The density is a product of 1-D factors, so it has $2^m$ equally weighted modes.
- The parameter $\delta$ controls mode separation, and so how hard it is to move mass between wells.

Settings in Section 4.1, both with $\delta = 2$:

| $d$ | modes | implied $m$ |
|---|---|---|
| 10 | 8 | 3 |
| 50 | 32 | 5 |

The paper states the mode counts, not $m$; $m$ follows from $2^m$. The appendix also uses $d = 10$, $m = 10$
(1024 modes) to compare baselines.

### 3.2 Why it suits TTFlow

- **Known structure.** The density factorises over coordinates, so its log-density has TT rank 1 and its
  score is separable. The rectified-flow velocity field $v^\star(x,t)$ is **not** separable in general: the
  coupling between $x_0$ and $x_1$ mixes coordinates. How much rank $v^\star$ needs is an open empirical
  question, and it is the question TTFlow asks.
- **Controlled difficulty.** We can change $d$, the number of modes $2^m$ and the separation $\delta$
  independently.
- **Exact references.** Exact samples are cheap (1-D rejection sampling per well) and $\log Z$ is known,
  so we can measure sample quality without a reference sampler.

### 3.3 What the paper measures

ESS, log-variance divergence, $\log Z$ and its relative error. Its main finding for Section 4.1 is that the
TT-HJB solver is stable from $d = 10$ to $d = 50$ and faster and more accurate than the neural DIS sampler.
These are **importance-weighted sampler** metrics. They need a density for the model, $\log \hat\pi_1(x)$,
which a rectified flow gives only through the instantaneous change-of-variables formula
(an ODE integral of $\nabla\!\cdot v_\theta$). See Section 5.

---

## 4. Our setup in this repo

### 4.1 Pipeline

```
data_snapshot_generator.py      ->  data-snapshots/*.pkl     (x1 samples: train/test/eval = 500k/200k/200k)
training/nn                     ->  models/*.pkl              (VelocityFieldRegressionNet, MLP width 128, depth 2, Adam)
inference/inference_and_evaluation.py
                                ->  generated samples, trajectory plots, Sinkhorn evaluation
```

- Source $\pi_0 = \mathcal N(0, I_d)$, random pairing (1-RecFlow), then one reflow (2-RecFlow).
- Global velocity model, one network for all $t \in [0,1]$.
- ODE solvers: `euler`, `rk4`, `dopri8`, each with 1000 time points.
- Metric: debiased Sinkhorn divergence (`geomloss`, $p=2$) over a blur grid. Each value is reported
  against the **real–real noise floor**, so a perfect generator would match the floor. See
  [`experiments.md`](../experiments.md) for the protocol.

### 4.2 ⚠️ The original `many_well` target is not TTD Eq. (18)

`ManyWell_Gaussian` in
[`ttflow/data_generation/distributions/many_well.py`](../ttflow/data_generation/distributions/many_well.py)
implements the **FAB / Midgley et al. (2022) Many-Well**, not TTD Eq. (18):

| | TTD Eq. (18) | Repo `ManyWell_Gaussian` |
|---|---|---|
| 1-D well | $(x^2-\delta)^2$, symmetric | $x^4 - 6x^2 - 0.5x$, **asymmetric** (`a=-0.5, b=-6, c=1`) |
| Mode locations | $\pm\sqrt\delta = \pm 1.41$ ($\delta=2$) | about $\pm 1.7$ |
| Mode weights | equal | **unequal**: the deep well holds more mass |
| Structure | $m$ well coords + $(d-m)$ Gaussian coords | pairs (well, Gaussian) repeated `many_well_dim/2` times, then `total_dim − many_well_dim` Gaussian coords |
| Modes | $2^m$ | $2^{\texttt{many\_well\_dim}/2}$ |

So the current runs are:

| Experiment | `total_dim` | `many_well_dim` | double-well coords | modes |
|---|---|---|---|---|
| [#12](../experiments_and_results/12/desc.md) | 8 | 4 | 2 | 4 |
| [#13](../experiments_and_results/13/desc.md) | 8 | 8 | 4 | 16 |

Both targets are fine benchmarks, and the unequal weights make the FAB version a harder test of mode
weighting. But **the numbers from #12 and #13 cannot be compared with the TTD paper.**

### 4.3 The Eq. (18) target: `multiwell_ttd`

[`MultiwellTTD`](../ttflow/data_generation/distributions/multiwell_ttd.py) implements TTD Eq. (18) exactly:

- **Exact sampling.** Each well coordinate uses rejection sampling from a two-Gaussian proposal centred at
  $\pm\sqrt\delta$. The other coordinates are drawn from $\mathcal N(0,1)$.
- **Exact $\log Z$:** $\;m\log Z_\delta + \tfrac{d-m}{2}\log 2\pi$, where $Z_\delta = \int e^{-(x^2-\delta)^2}\mathrm dx$
  is computed by quadrature.
- **Closed-form `score(x)`.**
- **Mode diagnostics.** `mode_occupancy(x)` gives the fraction of samples in each of the $2^m$ sign patterns.
  `mode_occupancy_tv(x)` gives the total-variation distance from the uniform $2^{-m}$ weights.
  This answers point 1 of Section 5 directly.

Generate a snapshot (paper settings: `--multiwell-dim 50 --multiwell-m 5`, or `--multiwell-dim 10 --multiwell-m 10`):

```bash
PYTHONPATH=. python ttflow/data_generation/data_snapshot_generator.py --dataset-name multiwell_ttd --multiwell-dim 10 --multiwell-m 3 --multiwell-delta 2.0
```

Sanity checks (200k samples, $\delta = 2$):

| Check | Result |
|---|---|
| $\mathbb E[x_1^2]$, empirical vs. exact | 1.836 vs. 1.835 |
| Normalised density integrates to 1 ($d=2$ grid) | 0.99999999 |
| Closed-form score vs. autograd | max error 0 |
| Mode-occupancy TV, $d=10, m=3$ | 0.003 (sampling noise) |
| $\log Z$, $d=10, m=3$ / $d=50, m=5$ / $d=10, m=10$ | 7.312 / 42.817 / 2.930 |

Next step: train 1-RecFlow and 2-RecFlow on $d=10, m=3, \delta=2$, the sanity check from the README roadmap.

### 4.4 Example output (Experiment #13, FAB target, 1-RecFlow, RK4)

![Trajectories, Experiment 13, 1-RecFlow, RK4](../experiments_and_results/13/1-RecFlow/rk4/trajectory_many_well_VelocityFieldRegressionNet_num_reflows_0_many_well_total_data_dim=8,%20many_well_dim=8_rk4_2026-11-19T15:08:04.736096.png)

<!-- TODO: add a short table of Sinkhorn real–generated vs real–real per solver and per reflow for #12 and #13 -->

---

## 5. Things to watch on multiwell targets

1. **Mode weights, not just mode coverage.** A flow can reach every well and still put the wrong mass in
   each. On the asymmetric FAB target, compare the fraction of samples per well with the exact weights. On
   Eq. (18), each of the $2^m$ modes should hold $2^{-m}$ of the mass.
2. **Mode bridges.** With random pairing, mean regression smooths the velocity between wells, which can
   leave a thin trail of samples in the low-density saddle. Reflow usually reduces this, but it can also lock
   in the teacher's mode-weight errors, because the student only sees the teacher's endpoints.
3. **Curvature and solver choice.** On multimodal targets the 1-RecFlow field changes sharply in time near
   the point where paths "decide" which well to enter. That is where Euler loses accuracy and RK4 or dopri8
   (or the blog post's time warp) helps. After reflow the gap between solvers should shrink. If it doesn't,
   the reflow did not straighten the paths enough.
4. **Metric sensitivity.** At small blur, Sinkhorn behaves like $W_2$ and responds to mode weights. At
   large blur it only sees the overall mean and spread. Report the gap to the real–real floor, not the raw value.
5. **Comparability with TTD.** To report ESS or $\log Z$ as TTD does, we need $\log\hat\pi_1$:
   $\log\hat\pi_1(\hat x_1) = \log\pi_0(x_0) - \int_0^1 \nabla\!\cdot v_\theta(x_t,t)\,\mathrm dt$,
   with a Hutchinson trace estimator for larger $d$. This is worth adding once the Eq. (18) target exists.

---

## 6. Link to the TTFlow plan

The NN teacher trained here provides:

- **trajectories** $\{x_t\}$ to split into $K$ time segments (README roadmap, step 2);
- **regression targets** $v_\theta(x_t,t)$ for fitting FTT students per segment;
- a **baseline** for sample quality, memory and training time.

Multiwell is the first target because its structure is known and its difficulty can be tuned. If an FTT
velocity field with modest rank cannot match the NN teacher here, it will not match it on
$\phi^4$ / Ginzburg–Landau (TTD Section 4.2), where coordinates are coupled.

---

## References

- *Flow With What You Know.* ICLR 2025 Blogposts Track, 2025. <https://iclr-blogposts.github.io/2025/blog/flow-with-what-you-know/>.
  The post has no named authors; cite it by title and URL.
- R. Gruhlke, J. Berner, D. Sommer, L. Richter. *Tensor Train Diffusion: Leveraging Low-Rank Structures for
  High-Dimensional Score-Based Sampling.* ICML 2026. [arXiv:2607.06841](https://arxiv.org/abs/2607.06841).
- X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003).
- Y. Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- L. Midgley et al. *Flow Annealed Importance Sampling Bootstrap.* ICLR 2023. [arXiv:2208.01893](https://arxiv.org/abs/2208.01893). This is the source of the FAB Many-Well target used in the repo.
