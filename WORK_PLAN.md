# TTFlow Work Plan

All planning for TTFlow lives in this file.

**How to use it**

- Each item has a short title, its deliverable, and a ticket (GitHub issue) once one is opened: `— [#N](https://github.com/mbaddar1/ttflow/issues/N)`.
- When an item is finished, tick it, **move it to DONE** with the completion date, and link the result:
  a commit, an `experiments.md` entry or a `docs/` article.
- Keep TODO in priority order, top item first.

Tickets: <https://github.com/mbaddar1/ttflow/issues>

---

## TODO

### 1. Blog post 01: Rectified Flow on the ManyWell Problem, an exploratory experiment — _ticket: TBD_

- [ ] Write blog post 01, *Rectified Flow on the ManyWell Problem: An Exploratory Experiment*, using the ICLR 2025 blog post [*Flow With What You Know*](https://iclr-blogposts.github.io/2025/blog/flow-with-what-you-know/) as the guide. A draft already exists.
- [ ] Use the current `many_well` target with results from experiments #12 and #13: 1-RecFlow vs. 2-RecFlow, ODE solvers, Sinkhorn results table, figures.
- [ ] **What is different from *Flow With What You Know*:**
  - **Higher-dimensional data:** the guide post uses 2-D toy targets (Gaussian → spiral). This post runs on the ManyWell target in higher dimensions ($d = 8, 16, 32$).
  - **Expressive power vs. generation quality and inference speed:** measure how the expressive capacity of the velocity network (width and depth of `VelocityFieldRegressionNet`) affects the quality of the generated distribution and the inference speed. _Metrics and setup: TBD._
- **Deliverable:** [`docs/01-rectified-flow-multiwell-exploratory.md`](docs/01-rectified-flow-multiwell-exploratory.md)

### 2. Blog post 02: PeRFlow, read and explain the paper — _ticket: TBD_

- [ ] Read the first 6 pages of PeRFlow ([Yan et al., 2024, *PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator*](https://arxiv.org/abs/2405.07510)).
- [ ] Write a clear, documented account of **the value the paper introduces**: what problem it solves, what is new compared with rectified flow and reflow, and why it matters for inference.
- **Deliverable:** `docs/02-piecewise-linearization-for-flow-models-an-inference-optimization-perspective.md`
  (*Piecewise Linearization for Flow Models: An Inference Optimization Perspective*)

### 3. Experiment 4: PeRFlow segmentation, NN teacher → smaller NN student — _ticket: TBD_

- [ ] Implement the PeRFlow piecewise reflow loss: split $[0,1]$ into $K$ segments and straighten the flow inside each segment, using teacher trajectories.
- [ ] **Teacher:** one pretrained NN rectified flow per dimension (`VelocityFieldRegressionNet`, expressive: wide and deep).
- [ ] **Student:** a smaller `VelocityFieldRegressionNet` (fewer and narrower layers) that approximates the teacher's segment trajectories.
- [ ] **Dataset:** fixed to the current `many_well` target (`ManyWell_Gaussian`, the FAB Many-Well used in experiments #12 and #13), with total dimension $d \in \{8, 16, 32\}$ to study how the results scale with dimension. Choose `many_well_dim` per $d$ (number of modes $= 2^{\texttt{many\_well\_dim}/2}$).
- [ ] **Experiment dimensions:**
  - How small can the student be relative to the teacher?
  - How do **inference quality** and **inference speed** change with **student size** and **number of segments $K$**?
  - How do these trade-offs change with **dimension** $d = 8, 16, 32$?
- **Deliverable:** code plus results in `experiments_and_results/` (Experiment 4).

### 4. Document Experiment 4 in `experiments.md` — _ticket: TBD_

- [ ] Write the Experiment 4 entry in [`experiments.md`](experiments.md): setup, quality vs. size vs. $K$ tables, speed results, figures and conclusions.

### 5. Experiment 5: PeRFlow segmentation, NN teacher → FTT student — _ticket: TBD_

- [ ] Same teachers, datasets ($d \in \{8, 16, 32\}$) and PeRFlow segment loss as Experiment 4, but the student is a **functional tensor train (FTT)**.
- [ ] Compare two architectural paradigms:
  - **(a) Separate FTTs**, one per segment.
  - **(b) One FTT with time as an input**, similar to the PeRFlow paper.
- [ ] Use the best NN student from Experiment 4 as the baseline (quality, speed, number of parameters).
- **Deliverable:** code plus results in `experiments_and_results/` (Experiment 5).

### 6. Document Experiment 5 and ALS / DMRG convergence — _ticket: TBD_

- [ ] Write the Experiment 5 entry in [`experiments.md`](experiments.md).
- [ ] Report how **ALS** and **DMRG** converge on this specific problem: loss vs. sweeps, rank behaviour, runtime and stability, for both paradigms (a) and (b).

### Later: more expressive NN architectures — _ticket: TBD_

- [ ] Repeat Experiments 4 and 5 with a more sophisticated velocity network (e.g. U-Net) as teacher and/or NN student.
  All NNs in Experiments 4 and 5 are the simple `VelocityFieldRegressionNet`.

---

## DONE

- [x] Create this work plan as the single planning file (2026-11-25) — [WORK_PLAN.md](WORK_PLAN.md)
- [x] Make `many_well` snapshot generation work at $d = 32$: `pairwise_plot` now accepts any $d \ge 2$ (2026-11-25) — [many_well.py](ttflow/data_generation/distributions/many_well.py)
- [x] Add the TTD Eq. (18) Multiwell target `multiwell_ttd` (2026-11-25) — [multiwell_ttd.py](ttflow/data_generation/distributions/multiwell_ttd.py)
