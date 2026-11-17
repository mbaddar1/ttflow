# TT-Flow: Adaptive Low-Rank Rectified Flows

## Overview
Rectified flows are a special type of generative models which try to solve the following problem : 
Given a pairing of source and target sample $x_0\sim \pi_0 ,x_1\sim \pi_1$ , the trajectory of generation from $x_0$ to
$x_1$ can be models as an ODE:

$\hat{x_1}=x_0+\int_{t=0}^{1}v(x_t,t)dt,$

where the generated sample $\hat{x_1}$ follows the push forward distribution $\hat{\pi}$
and the samples $x_0,x_1,\hat{x_1} \in \mathbb{R}^d$ are $d$-dimensional vectors


*TT-Flow* is a research project that explores leverage low-rank Functional tensor trains for mode
## Development Work

* Main branch: `main`
* 

---

## Roadmap

### Phase 1: Distillation & Piecewise Linearization (FTT + DMRG)

* **Goal:** Distill teacher trajectories into adaptive low-rank FTT representations using DMRG rank adaptation.
* **Key Experiments:**
  1. **Trajectory Straightness Benchmark:** Measure local trajectory curvature and straightness across time segments $t \in [t_k, t_{k+1}]$.
  2. **Global vs. Local FTTs:** Compare a single unified continuous-time model $FTT(z,t)$ against decoupled, segment-specific FTT models.

### Phase 2: TT-Flow from Scratch

* **Goal:** Train a standalone generative Flow Matching model from scratch using FTT velocity cores, without a teacher model.
* **Key Focus:** Optimal Transport (Sinkhorn) pairing to resolve trajectory crossings, combined with mini-batch DMRG/ALS optimization sweeps.

---

## Possible Contributions

* **Adaptive Local Approximation:** Instead of fighting non-linearity through additional expensive global Reflow training cycles, TT-Flow accepts residual non-linearity and models it locally within small time windows $[t_k, t_{k+1}]$ using rank-adaptive tensor cores.
* **Entropy-Guided Residual Modeling:** Investigate whether entropy-guided rank adaptation can allocate representational capacity where local trajectory complexity demands it, connecting adaptive model structure with established ideas in low-rank tensor compression.
* **From Distillation to Generation:** Explore a path from compressing existing flow trajectories to learning generative velocity fields directly through tensor-train representations.

These are research hypotheses to validate experimentally, not assumed results.

---
# Experiments
## Settings 
Quoted from the Rectified flow paper , p23 the `experimental settings section`
* Using $\pi_0$ as Gaussian
* We use MLP and U-Net (with possible adaption to tabular data ??)
* Solvers : euler, rk4 and dopri8

## Experiment 1



___
## Key takeaways and observations 
* 


## References

1. **Local Flow Matching Generative Models.**  
   https://arxiv.org/abs/2410.02548
2. **PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator.**  
   https://arxiv.org/abs/2405.07510
3. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.**  
   https://arxiv.org/abs/2209.03003
4. **On the Convergence and Straightness of Rectified Flow**
    https://arxiv.org/pdf/2410.14949
5. **Simple ReFlow: Improved Techniques for Fast Flow Models** (ICLR 2025).  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/a95ba79ea3881ffd9fb8620e1a0d4b36-Abstract-Conference.html
6. **Analyzing and Mitigating Model Collapse in Rectified Flow Models** (2024).  
   https://arxiv.org/abs/2412.08175
7. **Improving the Training of Rectified Flows** (NeurIPS 2024).  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/7343a5c976f8399880b695267f1f9e9f-Abstract-Conference.html
8. **ProReflow: Progressive Reflow with Decomposed Velocity** (CVPR 2025).  
   https://openaccess.thecvf.com/content/CVPR2025/html/Ke_ProReflow_Progressive_Reflow_with_Decomposed_Velocity_CVPR_2025_paper.html
