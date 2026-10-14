# GraphNN: Janus visual graph memory

The user requested a working GraphNN repository and a local GUI like the
uploaded Janus Cabbage: load A and B, train, move a slider, inspect outputs.
The approved direction replaces image-pair fitting in a complex coordinate
MLP with an explicit byte-bounded graph written by a learned network.

## Mechanism

Python 3.10+; PyTorch, NumPy, Pillow; Tkinter GUI; CUDA when available and CPU
otherwise. Default images are 32x32 RGB, graph 8x8 with 8 channels. Each
undirected edge has a gain byte and a phase byte per channel: 1,792 bytes
for the complete two-image memory. Shared neural weights are reported
separately. The connectivity is fixed; edge values and write gates learn.

A CNN and context-dependent gated updater write A then B into the same
graph. Reads receive only this state plus a query. Query angle controls
the spatial/channel excitation via sine/cosine features, not a global
complex rotation. A fresh complex ping propagates through weighted edges;
real/imaginary responses at multiple depths feed an RGB decoder. The
transport is lossy and is not claimed unitary or an exact interferometer.

The direct/vector control uses the same writer architecture and byte
budget, treats the flattened stored scalars as a code, and decodes them
without propagation. Train its weights independently on identical batches.
Report shared parameter counts and timing; equal bytes do not imply equal
compute. No graph superiority is presupposed.

## Two workflows

The GUI fits its loaded pair like original Janus. This is a demonstration
of training and explicit memory dependence, not evidence of generalization:
the fitted shared weights can also contain pair information. A separate
headless trainer learns across independent procedural image pairs, evaluates
held-out pairs, and saves a reusable writer/reader. Loading that checkpoint
and encoding a new pair performs no gradient update. Natural photographs
are outside the pilot distribution.

At 0 degrees read A, at 90 read B. Intermediate queries have no supervised
physical meaning. Show target thumbnails, graph/vector answers, gain/phase
maps, endpoint reconstruction errors, byte counts, and training progress.
Offer generated demo images, load A/B, fit/pause, frozen encode, reset,
randomized/blank memory controls, export view, save/load memory.

Saved memory includes validated configuration, CPU model tensors, uint8
graph/vector state and metadata, but no source pixels. Use weights_only
checkpoint loading. A saved memory must render without its source images;
training after loading requires both images to be supplied again.

All Tk calls occur on the main thread. Background workers communicate
through a queue; a lock protects models from concurrent training, reads,
saves and loads. Prevent overlapping training jobs and image replacement
while a job is stopping. Closing the window signals workers and cancels
scheduled UI callbacks. Bad images/checkpoints produce actionable errors.

## Verification

Behavior tests cover quantized bytes, gradient flow, ordered writing,
question dependence, repeatable fresh reads, malformed state, checkpoint
round-trip without source images, training loss reduction, memory corruption,
and actual GUI load/train/stop/slider/save/load interactions under Xvfb.
Run a held-out procedural pilot with memory-shuffle and blank controls.
Publish reproducible receipts and images, with no universal capacity claim.
