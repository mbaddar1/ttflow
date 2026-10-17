# TT-Flow: Adaptive Low-Rank Rectified Flows

## Manifesto

TT-Flow explores a simple principle: **generative flow trajectories should be approximated adaptively, not forced into a globally simple representation.**

Rectified Flow aims to straighten generative trajectories, but residual non-linearity can remain even after reflow. Rather than relying exclusively on expensive global training cycles to eliminate this non-linearity, TT-Flow investigates whether Functional Tensor Trains (FTT) and Density Matrix Renormalization Group (DMRG) algorithms can represent and solve the problem locally.

The central hypothesis is that **adaptive low-rank tensor representations can capture local trajectory structure efficiently, providing a practical alternative to globally uniform model complexity.**

The project follows two complementary directions: distilling existing flow trajectories into adaptive piecewise approximations, and ultimately training standalone generative flow models directly in the tensor-train representation.

---

## Dev Work

* Main branch: `main`

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

## References

1. **Local Flow Matching Generative Models.**  
   https://arxiv.org/abs/2410.02548

2. **PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator.**  
   https://arxiv.org/abs/2405.07510

3. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.**  
   https://arxiv.org/abs/2209.03003
