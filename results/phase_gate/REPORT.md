# Phase computer — actual 896-byte state after retraining (2026-10-14)

**Completed** exploratory experiment, not a claim of new physics or replacement of Transformers. **Three independent training seeds 41/97/127; 500 steps each; 128 held-out paired image histories per seed**. Same supervised A→B→A′ rewrite objective and deterministic data generation as GraphNN PR #2. The original 1,792-byte checkpoints are kept frozen and reevaluated, rather than retrained. The new 896-byte models and wide gain-only model start from scratch.

The [predeclared protocol](../../docs/phase_computer_gate.md) predates inspection of full 500-step results. [Full per-seed JSON](pilot500/summary.json) includes each seed's ablations and comparison metadata. [Phase/gain/vector 896-byte training logs and checkpoint artifacts](https://github.com/anttiluode/GraphNN/actions/runs/38585101081); [1792-byte wide gain-only logs and checkpoint artifacts](https://github.com/anttiluode/GraphNN/actions/runs/38585441325).

## Held-out performance

| Model | Inference state bytes | Shared parameters | Both marks correctly read after A′ | B RGB MSE ↓ | B prediction drift ↓ | A RGB MSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **Phase-only graph, trained fresh** | **896** | 43,211 | **384/384 (100%)** | **0.019891** | **0.000681** | **0.017565** |
| Gain-only graph, same-sized | 896 | 43,211 | 337/384 (87.76%) | 0.061785 | 0.011864 | 0.048367 |
| Direct vector, same-sized | 896 | 43,211 | 376/384 (97.92%) | 0.044139 | 0.017199 | 0.023709 |
| Original complex graph, frozen PR #2 | 1792 | 44,099 | 384/384 (100%) | 0.020380 | 0.000847 | 0.018129 |
| Original direct vector, frozen PR #2 | 1792 | 44,099 | 380/384 (98.96%) | 0.027431 | 0.009546 | 0.018815 |
| Wide gain-only graph (16 channels), trained fresh | 1792 | 58,035 | 349/384 (90.89%) | 0.058475 | 0.011409 | 0.048864 |

Each pair's stored graph is quantized to physical **uint8** after every write. The missing gain or phase factor is a fixed readout constant, regenerated at query time. No hidden 1792-byte per-pair state in the phase-only branch. **Shared weights and inference activations are not counted in the state bytes**; they are separately reported, and the graph reader costs more propagation compute than the vector.

## Findings that survive the stronger tests

1. **Phase-only can discard all 896 stored gain bytes and retrain to recover both marks perfectly** in this task, with excellent B retention and slightly better mean RGB error than the original complex 1792-byte graph. The small differences between phase-only and the full graph are not grounds to claim general superiority with only three seeds.
2. The same-sized **896-byte gain-only graph loses** (87.76% both-mark recovery and 0.0618 B RGB MSE), despite identical neural parameter count and optimization schedule.
3. Adding **twice as many gain channels (1792 bytes, 58,035 parameters)** improves joint mark recovery to 90.89%, but still falls short of the smaller phase-only model.
4. The 896-byte vector control gets 97.92% both-mark accuracy but much worse full-frame B MSE and B output drift. The older 1792-byte vector model is also inferior to the phase-only model on these B metrics.
5. Previous unchanged-A sham diagnostic shows broad memory-byte changes can arise simply from rewriting A. Low drift is evidence of readout insensitivity, not of precise local gauge/holonomy compensation.

## What we have NOT isolated

- **Positive versus signed edge weights**: gain-only uses nonnegative amplitudes, while phase allows sign reversal and complex rotations. A separately trained 896-byte signed-real transport control is necessary before attributing gains specifically to continuous U(1) interference rather than ordinary signed mixing.
- GraphNN transport is **dissipative**. Its Hermitian edge convention does *not* imply a unitary operator across its four propagation steps.
- Nothing enforces gauge invariance or measures closed-loop flux; [Wednesday](https://github.com/anttiluode/Wednesday) demonstrates an algebraic closed-loop invariant in a separate model.
- Only two 32×32 procedural images, one binary 6×6 patch per image and a single targeted rewrite. This does not demonstrate capacity under arbitrary histories, natural-photo fidelity, visual scene understanding, lifelong memory, or superiority to attention.
- The full baseline weights come from a prior PR #2 run with otherwise matching data/protocol, not the same freshly initialized training runs as the 896-byte variants. All five new runs were performed within the current gate.
- 384 queries represent only **three separately trained models**, not 384 independent training replications.

## Run it yourself

```bat
python3.13 -m pip install -r requirements.txt
python3.13 -m unittest discover -s tests -v
python3.13 train_phase_gate.py --steps 500 --seeds 41 97 127 --device cpu --output runs/phase_gate
python3.13 train_wide_gain.py --steps 500 --seeds 41 97 127 --device cpu --output runs/wide_gain_gate
python3.13 phase_computer_gui.py --seed 41 --runs runs/phase_gate
```

The GUI displays the original target alongside all five frozen readers, an A/B query-angle control, held-out sample selection, and before/after A′ rewrite. Alternatively, download the trained 896-byte checkpoints from the successful [Actions artifacts](https://github.com/anttiluode/GraphNN/actions/runs/38585101081) and extract each `seedN/trained_models.pt` under `runs/phase_gate`.

## Suggested decisive follow-up

Train a **same-byte, same-parameter signed-real graph** whose couplings can be positive or negative but cannot rotate by arbitrary complex angles. Compare it against the present phase-only graph, then examine trained phase winding / loop observables and update sensitivity. That discriminates a genuine continuous-phase computational advantage from a simpler advantage of allowing signed couplings.
