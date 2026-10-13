

# Dev work
* main branch is  `main`


# TT-Flow: Adaptive Low-Rank Rectified Flows

## Overview
**TT-Flow** explores Functional Tensor Trains (FTT) and Density Matrix Renormalization Group (DMRG) algorithms to compress and straighten generative flow trajectories in Rectified Flow models.

---

## Roadmap

### Phase 1: Distillation & Piecewise Linearization (FTT + DMRG)
* **Goal:** Distill teacher trajectories into adaptive low-rank FTT representations using DMRG rank-adaptation.
* **Key Experiments:**
  1. **Trajectory Straightness Benchmark:** Measure local trajectory curvature/straightness across time segments $t \in [t_k, t_{k+1}]$.
  2. **Global vs. Local FTTs:** Compare a single unified continuous-time model $FTT(z, t)$ against decoupled segment-specific FTT models.

### Phase 2: TT-Flow from Scratch
* **Goal:** Train a standalone generative Flow Matching model from scratch using FTT velocity cores without a teacher model.
* **Key Focus:** Optimal Transport (Sinkhorn) pairing to resolve trajectory crossings, combined with mini-batch DMRG/ALS optimization sweeps.

## Possible contribution 
* Instead of fighting non-linearity by forcing additional expensive global Reflow training cycles, TT-Flow accepts residual non-linearity and handles it locally inside small time windows $[t_k, t_{k+1}]$ using rank-adaptive tensor cores.