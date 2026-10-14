"""Checkpoint-only causal diagnostics for GraphNN PR #2.

Never optimizes weights. Reports large-displacement readout drift, matched
random perturbations, a SAME-A sham write, direct JVP, and phase/gain ablations.
Phase removal is an inference ablation, NOT a trained gain-only baseline.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from graphnn.data import procedural_pairs
from graphnn.rewrite import make_rewrite_batch, PALETTE
from graphnn.training import PairSession
from graphnn.model import GraphMemory

ROOT = Path(__file__).resolve().parents[1]
DIMS = (1, 2, 3)

def mse(a, b):
    return (a - b).square().mean(DIMS)

def norm(a):
    return a.flatten(1).norm(dim=1)

def memory(state, cfg):
    return GraphMemory.from_flat(state, cfg)

def memory_float(state):
    return state.flat().float() / 255.0 if state.flat().dtype == torch.uint8 else state.flat()

def quantize(x):
    return x.clamp(0, 1).mul(255).round().div(255)

def ablate(state, cfg, kind, perms=None):
    parts = list(memory(state, cfg).parts)
    if kind == "phase_removed":
        for i in (1, 3):
            parts[i] = torch.full_like(parts[i], 128.0/255.0)
    elif kind == "gain_fixed":
        for i in (0, 2):
            parts[i] = torch.full_like(parts[i], 128.0/255.0)
    elif kind == "edge_assignment_shuffle":
        assert perms is not None
        parts = [p.flatten(1)[:, ix].reshape_as(p) for p, ix in zip(parts, perms)]
    else:
        raise ValueError(kind)
    return GraphMemory(tuple(parts))

def random_equal_norm(before, delta, gen):
    """Convex bounded random target; report if the farthest-endpoint fallback is needed.

    Comparison is rounded to 8-bit; exact norm cannot always match to the
    last bit after rounding. Report observed norm ratio.
    """
    wanted = norm(delta)
    candidate = torch.rand(before.shape, generator=gen)
    raw = candidate - before
    available = norm(raw)
    fallback = available < wanted
    far = (before < 0.5).to(before.dtype)
    raw = torch.where(fallback[:, None], far - before, raw)
    available = norm(raw).clamp_min(1e-12)
    step = (wanted / available).clamp(max=1.0)
    out = quantize(before + step[:, None] * raw)
    ratio = norm(out - before) / wanted.clamp_min(1e-12)
    return out, ratio, fallback

def mark_hit(read, mask, labels, palette):
    mean_color = (read * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    guesses = (mean_color[:, None, :] - palette[None, :, :]).square().sum(-1).argmin(-1)
    return (guesses == labels).float()

def exact_jvp(model, cfg, at, displacement, angle):
    """Evaluate J_R(at) @ displacement with PyTorch forward/reverse AD."""
    def reader(flat):
        return model.read(memory(flat, cfg), angle)
    with torch.enable_grad():
        _, product = torch.autograd.functional.jvp(reader, (at.detach(),),
                                                    (displacement.detach(),),
                                                    create_graph=False, strict=False)
    return product.detach()

def evaluate_seed(seed, count, chunk, repeat, jvp_cases):
    checkpoint = ROOT / "results/rewrite_gate/pilot500" / f"seed{seed}" / "memory.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(2)
    session = PairSession(device="cpu", seed=seed)
    session.load(checkpoint)
    batch = make_rewrite_batch(procedural_pairs(count, seed=seed+10000),
                               seed=seed+20000)
    observations = {}
    for arm, model in session.models.items():
        model.eval()
        cfg = model.config
        palette = batch.original.new_tensor(PALETTE)
        gen = torch.Generator().manual_seed(890007 + seed*11 + (arm=="vector")*100003)
        sums = defaultdict(float)
        counts = defaultdict(int)
        jvp_left = jvp_cases
        with torch.no_grad():
            for start in range(0, count, chunk):
                end = min(count, start+chunk)
                a = batch.original[start:end, 0]
                b = batch.original[start:end, 1]
                updated = batch.updated[start:end, 0]
                mask_a = batch.masks[start:end, 0]
                newlabel = batch.new_a_labels[start:end]
                bs = len(a)
                first = model.write(a, 0.).quantized()
                prior = model.write(b, 90., first).quantized()
                revised = model.write(updated, 0., prior).quantized()
                sham = model.write(a, 0., prior).quantized()
                x = memory_float(prior)
                y = memory_float(revised)
                sh = memory_float(sham)
                d = y-x
                output_b0 = model.read(prior, 90.)
                output_b1 = model.read(revised, 90.)
                output_bsh = model.read(sham, 90.)
                output_a0 = model.read(prior, 0.)
                output_a1 = model.read(revised, 0.)
                def add(name, vector):
                    sums[name] += float(vector.sum())
                    counts[name] += len(vector)
                add("b_drift_actual", mse(output_b1, output_b0))
                add("b_drift_sham", mse(output_bsh, output_b0))
                add("b_drift_detail_only_vs_sham", mse(output_b1, output_bsh))
                add("a_drift_actual", mse(output_a1, output_a0))
                add("b_reconstruction_mse_after", mse(output_b1, b))
                add("b_reconstruction_mse_before", mse(output_b0, b))
                add("b_mark_retained", mark_hit(output_b1, batch.masks[start:end, 1], batch.labels[start:end, 1], palette))
                add("a_updated_mark", mark_hit(output_a1, mask_a, newlabel, palette))
                add("changed_byte_frac_actual", (x!=y).float().mean(-1))
                add("changed_byte_frac_sham", (x!=sh).float().mean(-1))
                add("fraction_l2_due_to_sham", norm(sh-x)/norm(d).clamp_min(1e-12))
                add("actual_l2_norm", norm(d))
                # The exact same memory state is interrogated for every control.
                rand_for_jvp = None
                for j in range(repeat):
                    ctrl, ratio, fallback = random_equal_norm(x, d, gen)
                    if j == 0:
                        rand_for_jvp = ctrl-x
                    b_rand = model.read(memory(ctrl,cfg), 90.)
                    a_rand = model.read(memory(ctrl,cfg), 0.)
                    add("b_drift_random_normmatched", mse(b_rand, output_b0))
                    add("a_drift_random_normmatched", mse(a_rand, output_a0))
                    add("b_reconstruction_mse_random", mse(b_rand,b))
                    add("random_actual_to_target_norm_ratio", ratio)
                    add("random_fallback_fraction", fallback.float())
                # Apply identical edge-coordinate ablation on BEFORE and AFTER.
                perms = [torch.randperm(p[0].numel(), generator=gen)
                         for p in memory(x, cfg).parts]
                for kind in ("phase_removed","gain_fixed","edge_assignment_shuffle"):
                    mem0 = ablate(x, cfg, kind, perms)
                    mem1 = ablate(y, cfg, kind, perms)
                    b_before = model.read(mem0,90.)
                    b_after = model.read(mem1,90.)
                    a_before = model.read(mem0,0.)
                    a_after = model.read(mem1,0.)
                    add(kind+"_b_drift",mse(b_after,b_before))
                    add(kind+"_a_drift",mse(a_after,a_before))
                    add(kind+"_b_mse",mse(b_after,b))
                    add(kind+"_a_mark",mark_hit(a_after,mask_a,newlabel,palette))
                n = min(jvp_left,bs)
                if n:
                    # Full JVP for the two query directions; only a subset for speed.
                    # Exiting no_grad is handled inside exact_jvp.
                    da = d[:n]
                    dr = rand_for_jvp[:n]
                    for name,move in (("actual",da),("random",dr)):
                        jb = exact_jvp(model,cfg,x[:n],move,90.)
                        ja = exact_jvp(model,cfg,x[:n],move,0.)
                        add("local_jvp_b_"+name,mse(jb,torch.zeros_like(jb)))
                        add("local_jvp_a_"+name,mse(ja,torch.zeros_like(ja)))
                    jvp_left -= n
        observations[arm] = {k:round(sums[k]/counts[k],9) for k in sorted(sums)}
        observations[arm]["jvp_cases"] = min(jvp_cases,count)
        observations[arm]["per_arm_bytes"] = cfg.payload_bytes
    return observations

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    p.add_argument("--cases",type=int,default=128)
    p.add_argument("--chunk",type=int,default=16)
    p.add_argument("--random-repeats",type=int,default=4)
    p.add_argument("--jvp-cases",type=int,default=16)
    p.add_argument("--output",type=Path,default=Path("runs/diagnostic_rewrite.json"))
    args=p.parse_args()
    results={}
    for seed in args.seeds:
        results[str(seed)]=evaluate_seed(seed,args.cases,args.chunk,
                                        args.random_repeats,args.jvp_cases)
        print("SEED_RESULT_JSON="+json.dumps({"seed":seed,"metrics":results[str(seed)]},sort_keys=True),flush=True)
    means={}
    for arm in ("graph","vector"):
        keys=[k for k in results[str(args.seeds[0])][arm] if isinstance(results[str(args.seeds[0])][arm][k],float)]
        means[arm]={k:round(sum(results[str(s)][arm][k] for s in args.seeds)/len(args.seeds),9) for k in keys}
    receipt={"method":"checkpoint intervention; no retraining", "seeds":args.seeds,
             "cases_per_seed":args.cases,"random_perturbations_per_case":args.random_repeats,
             "jvp_cases_per_seed":args.jvp_cases,"per_seed":results,"mean":means,
             "caveats":[
                "Same original PR2 checkpoints and evaluation image generator; never optimizes weights.",
                "Sham writes full original A again; therefore it tests ordinary overwriting, not a pixel-free patch command.",
                "Matched random comparison is bounded, rounded-to-byte and compared at measured actual norms.",
                "JVP is local derivative at old state along the large finite memory displacement, not global nonlinear invariance.",
                "Zeroing phases / fixing gains / shuffling edge assignments at inference is distribution shift, NOT retraining controls.",
                "Topological loop holonomy is not measured or claimed; no causal claim of compensation from output drift alone."
             ]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("MEAN_RESULT_JSON="+json.dumps(means,sort_keys=True),flush=True)
    print("WROTE="+str(args.output),flush=True)

if __name__=="__main__":
    main()
