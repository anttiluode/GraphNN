# GraphNN Rewrite Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Execute tasks in order with failing tests before implementation.

**Goal:** Measure and expose A → B → revised A memory updates.
**Architecture:** Reuse JanusModel.write/read and explicit byte states; add
rewrite data/loss/metrics, true single-write sessions and existing GUI controls.
**Tech Stack:** Python, PyTorch, NumPy, Pillow, Tkinter, unittest.
**Spec:** docs/superpowers/specs/2026-10-14-rewrite-gate-design.md

## Global Constraints

- 1,792-byte memory and 44,099 shared parameters per default arm unchanged.
- The third write receives updated A and previous state only.
- Frozen writes do not change shared weights. Quantize every write.
- No source pixels in checkpoints; old format remains compatible.
- Training: 500 steps, batch8, train128, test128, seeds41/97/127, lr0.003.

## Review Focus

- Staging A must not silently re-encode B or change existing memory metrics.
- A write after source-free checkpoint loading must work without B's pixels.
- All masks/labels used for scoring stay out of reader features.
- No-write and shuffle controls must use the claimed state and target.
- GUI task selection and write completion must not access Tk from workers.

### Task 1: Rewrite examples, loss and true single-write sessions

**Files:** graphnn/rewrite.py, graphnn/training.py, tests/test_rewrite.py.
**Interfaces:** make_rewrite_batch(pair,seed); fit_rewrite_step(models,optimizers,batch); evaluate_rewrites(models,batch); PairSession.write_image(image,context), rewrite_a(), step(rewrite=False), set_pair(images,encode=True).

- [x] Add tests for localized flips, fixed weights, no B replay, staged targets, source-free writes and metric controls.
- [x] Run `python -m unittest discover -s tests -p test_rewrite.py -v`; expect missing rewrite module.
- [x] Implement deterministic marks, five-read training, state controls and session methods.
- [x] Run complete non-GUI suite; expect pass; commit.

### Task 2: Gate runner, GUI and measured delivery

**Files:** train_rewrite.py, graphnn/gui.py, tests/test_gui.py, README.md, results/rewrite_gate/.
**Interfaces:** use Task 1 functions; GUI job completion shows actual written targets and retained-B scores.

- [x] Add live-widget tests for staged sources, Write A/B and Rewrite A patch; observe failure.
- [x] Implement task selector, explicit writes, patch action and gate runner.
- [x] Run the frozen three-seed pilot and preserve all outcomes plus model weights/contact sheet.
- [x] Run full display-enabled tests, inspect GUI, obtain independent review and fix findings.

Delivery: publish this verified tree and bundled results through GitHub.
