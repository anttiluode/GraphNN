# Gate: Does a true half-budget phase graph work after training from scratch?

October 14, 2026. This protocol was written **before reading any full 500-step phase/gain training results**, but after the exploratory 1792-byte PR2 and checkpoint diagnostics that motivated it. It is a **predeclared follow-up**, not an independently preregistered study.

## Main question

PR2's 1792-byte graph stores edge gains and phases, and its **post-training phase-removal ablation** damaged readouts far more than clamping gains. That observation **does not prove** a phase-only model learns and retains at half the bytes. Here we actually fit one from scratch, and compare it with gain-only and vector competitors.

## Variants and budget

- `phase_only_896`: two uint8 planes, horizontal and vertical edge phases, `8 channels × 112 edges = 896 bytes`. Missing gain is the **fixed constant 0.5**, regenerated when reading. The original four-plane complex graph transport propagates queries through these edges.
- `gain_only_896`: identical half-plane writer/readout module and parameter count, but two uint8 planes now encode gains; phase is the exact **fixed neutral phase** 0.5. Graph transport is otherwise identical.
- `vector_896`: identical half-plane write architecture, parameter count and byte budget, with the original Janus direct-vector readout replacing graph propagation. Neither gain nor phase has privileged access at inference.
- `graph_1792_original`: the existing frozen PR2 graph checkpoint trained on the **same task, seed and training steps**.
- `vector_1792_original`: the corresponding existing frozen PR2 vector checkpoint.

The explicit half-memory `HalfMemory` has **exactly 896 stored uint8 values** and implements byte quantization between individual image writes. A temporary four-plane view in the reader recomputes omitted constant gain/phase factors: it is not an additional persistent cache. Readers may still use float/complex activations and learned shared parameters during inference; **896B is only the per-pair persistent payload, not total hardware RAM.** Report all trainable parameter counts; inference compute is *not* matched.

## Frozen data and score

From `train_rewrite.py` PR2: seeds 41/97/127, training 128 procedural A/B pairs each, test 128 disjoint pairs each, 32×32 RGB, two independently positioned blue/orange 6×6 marks, A mark flipped in A′. Train each new arm **500 steps**, Adam lr=0.003, batch size 8, with five supervised reads from A, AB, and A′B and identical full RGB plus masked-mark loss. After A→B, A′ is written with frozen weights using only revised A and the old state. B's pixels are never replayed to the writer.

Report per arm and per seed:
1. new A mark, retained B mark, *joint* mark classification accuracy;
2. B MSE (actual reconstruction) **and** B-output drift before/after A′;
3. A RGB MSE; ratio of changed stored bytes;
4. B-output drift of **unchanged A sham rewrite**, and no-write A′ mark;
5. parameter count and physical per-pair bytes;
6. shuffled-memory controls in which another example's memory replaces the read state.

A phase-only advantage requires consistent three-seed improvement in a relevant quality/retention metric **versus BOTH gain_only_896 and vector_896**, and no collapse in full RGB reconstruction. A storage-efficiency advantage against the full 1792B baseline requires near-matched RGB and mark quality with lower payload cost. Do not equate lower drift alone with better memory (a constant reader also has low drift). No result here establishes topology learning, physical unitary transport, loop holonomy, a natural-image generator or Transformer replacement.

## Why the prior Gemini explanation is not established

The original update is not a conservative unitary evolution: graphnn/model.py has a leaky mixture `0.5*field + 0.5*neighbors/degree`. The reverse Hermitian edge convention does not by itself make the entire propagation unitary or nonvanishing. Some learned phase behavior may simply reflect efficient signed modulation of spatial features. In the PR2 diagnostic, **rewriting an unchanged A alters about 85.5% of phase/gain bytes**, so byte changes cannot directly be interpreted as active loop compensation or biologically meaningful representational drift.

## Reproduction

```bash
python -m unittest discover -s tests -v
python train_phase_gate.py --steps 500 --seeds 41 97 127 --device cpu --output runs/phase_gate
```

The CI workflow trains seeds in independent jobs and publishes per-seed JSON and model checkpoint artifacts. Any pooled interpretation must distinguish trained seeds from 128 correlated queries within each seed.
