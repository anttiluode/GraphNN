# GraphNN PR #2: falsification of a compensating-loop rewrite story

**Status: completed checkpoint experiment.** [Successful GitHub Actions run](https://github.com/anttiluode/GraphNN/actions/runs/38580356963) · [Machine-readable raw results](pr2_checkpoint_probe.json) · [Diagnostic code](../../diagnostics/measure_rewrite_geometry.py).

Three originally trained seeds **41/97/127**, same **128 held-out pairs per seed**, same pre-trained checkpoint weights, **no retraining**. Four independent per-example norm-matched, bounded, 8-bit-rounded perturbations. True PyTorch automatic differentiation JVPs on **16 pairs/seed** (local gradients at before memory). Random perturbation norms matched to within approximately **0.02%**; no norm-fallback events.

## The decisive observations

| Measure (three-seed mean) | GraphNN | Equal-byte vector |
| --- | ---: | ---: |
| B output drift after actual A flip | **0.000847196** | 0.009546474 |
| B output drift after random state perturbation of same norm | 0.011544513 | **0.003613569** |
| B output drift after rewriting **unchanged A** | 0.000652411 | 0.001747676 |
| A output change after actual A flip | 0.023402282 | 0.019510764 |
| A output change after norm-matched random state perturbation | 0.008598360 | 0.008865545 |
| Changed bytes, actual update | 85.88% | 93.93% |
| Changed bytes, **unchanged-A sham** | **85.46%** | 93.12% |
| Local B Jacobian Jv squared, true rewrite direction | **0.001460538** | 0.013006256 |
| Local B Jacobian Jv squared, random matched direction | 0.010923701 | **0.002518983** |
| Local A Jacobian Jv squared, true direction | 0.005871704 | 0.005147028 |
| Local A Jacobian Jv squared, random direction | 0.007394826 | 0.006595557 |
| B RGB MSE after actual rewrite | **0.020380314** | 0.027430810 |

GraphNN's B readout moves **~13.6x less** for its actual rewrite than for a same-size random perturbation; its **local B Jacobian** is **~7.5x less sensitive** along the actual write displacement than along a random vector. This is evidence for learned query-selective insensitivity, not a proof of an exact or global nullspace. By contrast, the vector model's actual writer perturbation hurts B more than its random one. A's full nonlinear response changes substantially along the actual write, while the **local A JVP does not exceed the random control**. Therefore it would be misleading to claim a locally perfect A/B orthogonal decomposition.

**The key negative control:** writing A again without changing its detail changes **85.46% of graph bytes**, versus **85.88%** for the actual color flip. A broad distributed write alone cannot be interpreted as specialized compensating edge edits. The original writer was explicitly trained to preserve B after the A rewrite, so protection is a supervised behavior, not evidence of a surprising biological memory law.

## Test-time ablation of the trained representation

| GraphNN inference treatment | B reconstruction MSE after update | Updated A mark accuracy |
| --- | ---: | ---: |
| Normal trained phases and gains | **0.020380314** | **100%** |
| Replace all phase planes by neutral phase | 0.119621668 | 50.00% |
| Replace gains by midpoint | 0.023433165 | 100% |
| Shuffle edge assignments in both states | 0.105768565 | 50.78% |

Complex phase features and positional alignment are **necessary to this trained checkpoint** for the marked-detail task. But this does **not** isolate closed-loop flux or show that phase is necessary for an equally budgeted separately **retrained** model: any intervention of this size can put the decoder far out of distribution. With neutral phases the B output becomes largely unresponsive, so near-zero *drift* there is not evidence of protection—the B reconstruction MSE more than quintuples.

The actual GraphNN readout is dissipative, and neither its writer nor its loss enforces gauge invariance or Wednesday's compensated loop holonomy.

## Next narrowly targeted test

1. Retrain a phase-disabled graph and an equal-parameter gain-only graph under exactly PR #2's protocol. Compare *joint mark recovery, B drift and full-image quality* to the existing model on new seeds.
2. Separately assess **spatially correlated** and per-coordinate random memory perturbations; keep every perturbation's actual 8-bit norm and clipped fraction.
3. Measure loop flux shifts before/after rewrites and gauge rephasing of the trained edge states; test whether loop observables correlate with B stability, without assuming they are the stored representation.
4. Compare a trained rewrite with **unchanged A**, an A′ write, and incremental local patches while increasing number of alternating A/B rewrites.

Do not read the pilot as evidence of infinite capacity, a Transformer replacement or natural-image retention.
