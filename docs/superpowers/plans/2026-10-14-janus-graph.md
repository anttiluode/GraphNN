# Janus Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Deliver a runnable Janus-style graph-memory GUI and a measured reusable-memory pilot.

**Architecture:** A gated image writer creates an explicit quantized edge memory.
A question-conditioned complex propagation reader and a direct/vector control
produce RGB; a thread-safe session feeds a Tk GUI and saves pixel-free memories.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pillow, Tkinter, unittest.

**Spec:** docs/superpowers/specs/2026-10-14-janus-graph-design.md

## Global Constraints

- Default 32x32 RGB, 8x8 graph, 8 channels; 1,792 uint8 bytes per image pair.
- Reads use stored memory and query only. Saved memory contains no source pixels.
- CPU and automatic CUDA selection; all Tk operations on the main thread.
- Pair fitting and held-out reusable writer evaluation are explicitly different.
- Compare independent graph/vector models at equal stored bytes; report parameters.

## Review Focus

- Corrupt or incompatible checkpoints must leave the current session usable.
- Closing or pausing during a step must not trigger widget calls from a worker.
- Loading one image after a checkpoint must not silently refit shared weights.
- Memory controls must actually alter bytes used by the reader.
- Sliders at intermediate angles must not display a fictitious ground-truth metric.

### Task 1: Model and byte-memory contract

**Files:** graphnn/model.py, graphnn/__init__.py, tests/test_model.py.
**Interfaces:** ModelConfig; GraphMemory.quantized()/flat(); JanusModel.write(image, context, previous), encode_pair(pair), read(memory, angle), payload_bytes.

- [x] Write behavior tests for bytes, valid queries, writer gradients and fresh reads.
- [x] Run `python -m unittest discover -s tests -p test_model.py -v`; expect missing module.
- [x] Implement fixed lattice gain/phase memory, learned gated writing, graph and direct/vector readers.
- [x] Run the model tests; expect all pass; commit model and tests.

### Task 2: Training, portable saved memories, and pilot

**Files:** graphnn/data.py, graphnn/training.py, train.py, tests/test_training.py.
**Interfaces:** demo_pair(size, seed); procedural_pairs(count, size, seed); PairSession.set_pair(), step(), encode(), render(), save(), load(); fit_step(models, optimizers, pair); evaluate(models, pairs).

- [x] Write tests for improving pair loss, saved reads without images, bad loads and controls.
- [x] Run training tests; expect missing module.
- [x] Implement deterministic data, shared multi-pair trainer, thread-safe pair session and validated weights_only checkpoint loading.
- [x] Run full suite; expect pass; run a held-out pilot, save metrics/checkpoint/contact sheet; commit.

### Task 3: GUI and delivery

**Files:** graphnn/gui.py, janus_cabbage_graph.py, tests/test_gui.py, README.md, .github/workflows/tests.yml.
**Interfaces:** JanusApp(root, session); main(); main-thread queue polling and single training worker.

- [x] Write real-widget tests for slider, fitting/pause, save/load, error recovery and close.
- [x] Run GUI tests under Xvfb; expect missing GUI.
- [x] Implement the GUI, generated demo, controls, map display, export and documentation.
- [x] Run full tests, inspect an actual window screenshot, review the branch, commit and publish to the authorized repository.
