# GraphNN rewrite and retention gate

The user authorized continuing GraphNN after the proposed A → B → updated A
experiment. Add actual single-image writes to existing stored memory and make
the same operation usable in the GUI. Keep the 1,792-byte default state and
the matched graph/vector architectures unchanged.

## Contract

Write A at query identity 0°, write B at 90°, then write a revised A at 0°.
The third write receives only revised A, its identity and the previous byte
state. It must not rebuild from both source pictures, update shared weights,
or retain writer activations. The benchmark holds all shared weights fixed
while evaluating each sequence and quantizes after every write.

Procedural examples carry independently randomized blue/orange 6x6 marks on
A and B at independently randomized locations. A's last write flips its mark
and leaves all other pixels identical. Evaluator masks are never reader
inputs. Read A and B after the initial pair and after the update. Measure new
A mark recovery, B mark retention, joint success, RGB/mark MSE, B prediction
change and changed byte fraction. Include no-write, shuffled-final-memory,
and a privileged rebuild-with-B-pixels control. Report all of them.

Freeze the training comparison before running: 500 steps, batch 8, 128
training pairs, 128 held-out pairs, seeds 41/97/127, default architecture,
learning rate 0.003. Train the five reads after A, A→B and A→B→A' with
equal RGB-plus-mark losses in both arms. A mechanism passes when mean new-A
and retained-B mark accuracy both reach 90% and mean joint accuracy reaches
80%. A graph advantage requires better joint accuracy than the vector in
each seed; otherwise report not demonstrated. This is a short synthetic
rewrite gate, not arbitrary-history retention or photo generalization.

## GUI and sessions

Loaders stage source pictures. Explicit Write A/Write B update existing memory
with frozen weights; Rewrite A patch paints a contrasting mark and writes A
only. Frozen pair encode remains an explicit fresh A→B rebuild. Fit task can
select pair or rewrite training. Stored-target metrics describe the pictures
actually written, not staged replacements. Partial source-free checkpoint
sessions permit single writes and report only known target metrics. Saved
files remain pixel-free and old version-1 checkpoints remain readable.

All heavy writes/training use the existing job worker and model lock. UI
thumbnails, progress and write reports update on the main thread. Verify live
write/patch controls, staged-versus-written targets, source-free saved reads,
no parameter update during frozen writes, and no replay of B at an A rewrite.
