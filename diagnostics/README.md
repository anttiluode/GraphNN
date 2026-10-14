# Rewrite subspace diagnostic (GraphNN PR #2)

This diagnostic makes **no optimizer steps**. It loads the exact PR #2
`results/rewrite_gate/pilot500/seed{41,97,127}/memory.pt` models and rebuilds
the original held-out dataset with the same deterministic generator and masks.

Run locally:

```bash
python -m diagnostics.measure_rewrite_geometry --seeds 41 97 127 --cases 128 --random-repeats 4 --jvp-cases 16 --output runs/diagnostic-rewrite.json
```

It compares the reader's response before A is updated (M_AB) and after
the original prescribed A-detail flip (M_A'B). Outputs include:

- Mean B-output drift and A-output shift (squared RGB differences).
- Matched-*L2* random bounded perturbations of the **same prior memory**,
  rounded to uint8. The measured perturbation norm ratio and fallback rate
  are reported to audit the matching.
- A **sham rewrite** of unchanged full A from M_AB, and an additional comparison
  between the real and sham final B images. This distinguishes the patch flip
  from broad write-side effects.
- **True automatic-differentiation directional Jacobian products** for A and B
  with both actual and random displacements, at M_AB, on the first 16
  held-out examples per seed. A small local Jv does not prove a large
  displacement lies in a global nullspace.
- Phase-removed, gain-fixed and identical-edge-assignment-shuffle ablations
  applied to *both* old and updated states. These are severe, zero-retraining
  inference shifts, NOT a trained gain-only control nor an actual
  topology-learning experiment. Always report output quality as well as drift.
- The original binary mark accuracies, reconstruction MSE and number of
  changed bytes, for graph and matched vector readers.

Important: GraphNN does not implement Wednesday's exact loop-holonomy
invariants or explicitly perform local gauge-compensating writes. Merely
finding that B barely changes despite many altered bytes is not evidence
of an effective Jacobian nullspace or loop-phase protection. Interpretation
must depend on measured normalised random baselines, sham writes, and JVPs.

Equal *stored bytes* does not mean equal *reader computation*. Both arms
saw B-retention supervision during training, so preserved B is an expected
training target rather than a spontaneous discovery.
