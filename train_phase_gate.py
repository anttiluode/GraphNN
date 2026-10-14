"""Frozen GraphNN PR2 comparisons vs trained 896-byte phase/gain/vector models.

Explicitly count actual uint8 persistent bytes; fixed gain/phase readers do
not carry additional per-example state. The original 1792B PR2 checkpoint
models serve as frozen pre-recorded reference with identical train/eval data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from graphnn.data import procedural_pairs
from graphnn.rewrite import make_rewrite_batch, fit_rewrite_step, PALETTE
from graphnn.half_memory import HalfJanus, HalfMemory
from graphnn.training import PairSession


def write_states(m,batch):
    with torch.no_grad():
        a=m.write(batch.original[:,0],0.).quantized()
        ab=m.write(batch.original[:,1],90.,a).quantized()
        after=m.write(batch.updated[:,0],0.,ab).quantized()
        sham=m.write(batch.original[:,0],0.,ab).quantized()
    return ab,after,sham


def mark_correct(rendered, mask, labels, palette):
    mean=(rendered*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)
    chosen=((mean[:,None,:]-palette[None,:,:])**2).sum(-1).argmin(-1)
    return chosen.eq(labels).float().mean().item()


@torch.no_grad()
def evaluate(model,batch):
    model.eval()
    prior,after,sham=write_states(model,batch)
    outputs=[model.read(m,angle) for m in (prior,after,sham) for angle in (0.,90.)]
    pa,pb,aa,ab,sa,sb=outputs
    palette=batch.original.new_tensor(PALETTE)
    def err(a,b):
        return (a-b).square().mean().item()
    old=prior.flat()
    cur=after.flat()
    untouched=sham.flat()
    changed=(cur!=old).float().mean().item()
    changed_sham=(untouched!=old).float().mean().item()
    m={"a_mark":mark_correct(aa,batch.masks[:,0],batch.new_a_labels,palette),
       "b_mark":mark_correct(ab,batch.masks[:,1],batch.labels[:,1],palette),
       "a_rgb_mse":err(aa,batch.updated[:,0]),
       "b_rgb_mse":err(ab,batch.original[:,1]),
       "b_rgb_mse_before":err(pb,batch.original[:,1]),
       "b_drift":err(ab,pb),
       "b_sham_drift":err(sb,pb),
       "a_output_change":err(aa,pa),
       "changed_byte_fraction":changed,
       "sham_changed_byte_fraction":changed_sham,
       "no_write_new_a_mark":mark_correct(pa,batch.masks[:,0],batch.new_a_labels,palette),
       "payload_bytes":model.payload_bytes,
       "shared_parameter_count":sum(p.numel() for p in model.parameters())}
    masks=batch.masks
    colors=[]
    for output, mask, label in ((aa,masks[:,0],batch.new_a_labels),
                               (ab,masks[:,1],batch.labels[:,1])):
        average=(output*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)
        colors.append(((average[:,None,:]-palette[None,:,:])**2).sum(-1).argmin(-1).eq(label))
    m["joint_mark"]=torch.stack(colors).all(0).float().mean().item()
    shuffled=HalfMemory(tuple(p.roll(1,0) for p in after.parts))
    m["shuffled_b_mse"]=err(model.read(shuffled,90.),batch.original[:,1])
    m["shuffled_a_mark"]=mark_correct(model.read(shuffled,0.),masks[:,0],batch.new_a_labels,palette)
    return m


def train_seed(seed,args):
    torch.manual_seed(seed)
    models={}
    for kind,reader in (("phase_only","graph"),("gain_only","graph"),("phase_only","vector")):
        torch.manual_seed(seed)
        models[("vector_896" if reader=="vector" else kind+"_896")]=HalfJanus(kind=kind,reader=reader).to(args.device)
    optimizers={name:torch.optim.Adam(model.parameters(),lr=.003)
                for name,model in models.items()}
    # EXACT same procedural key and split as original PR2 train_rewrite.py.
    training=make_rewrite_batch(procedural_pairs(args.train_count,seed=seed),seed=seed+1000)
    test=make_rewrite_batch(procedural_pairs(args.test_count,seed=seed+10000),seed=seed+20000)
    rng=torch.Generator().manual_seed(seed)
    history=[]
    begun=time.perf_counter()
    for step in range(1,args.steps+1):
        ids=torch.randint(0,args.train_count,(args.batch_size,),generator=rng)
        batch=training.select(ids).to(args.device)
        losses=fit_rewrite_step(models,optimizers,batch)
        if step==1 or step%max(1,args.steps//5)==0 or step==args.steps:
            history.append({"step":step,**{k:round(v,7) for k,v in losses.items()}})
            print("TRAIN "+json.dumps({"seed":seed,"step":step,"loss":history[-1]}),flush=True)
    metrics={name:evaluate(model,test.to(args.device)) for name,model in models.items()}
    saved=PairSession(device=args.device,seed=seed)
    chk=Path(__file__).resolve().parent/"results"/"rewrite_gate"/"pilot500"/f"seed{seed}"/"memory.pt"
    if not chk.exists(): raise RuntimeError("Original PR2 checkpoint missing: "+str(chk))
    saved.load(chk)
    # Evaluated with identical endpoint data. Original models use 1792B.
    from graphnn.rewrite import evaluate_rewrites
    reference=evaluate_rewrites(saved.models,test.to(args.device))
    for k,v in reference.items():
        metrics[k+"_1792_original"]={
            "a_mark":v["after"]["a_mark_accuracy"],
            "b_mark":v["after"]["b_mark_accuracy"],
            "joint_mark":v["after"]["joint_mark_accuracy"],
            "a_rgb_mse":v["after"]["a_rgb_mse"],
            "b_rgb_mse":v["after"]["b_rgb_mse"],
            "b_drift":v["b_prediction_change_mse"],
            "changed_byte_fraction":v["changed_byte_fraction"],
            "payload_bytes":v["payload_bytes"],
            "shared_parameter_count":v["shared_parameters"]}
    out=args.output/f"seed{seed}"
    out.mkdir(parents=True,exist_ok=True)
    result={"seed":seed,"steps":args.steps,"train_count":args.train_count,
            "test_count":args.test_count,"batch_size":args.batch_size,
            "elapsed_seconds":time.perf_counter()-begun,
            "training":history,"evaluation":metrics,
            "limits":["Full baseline reuses frozen PR2 weights trained on the identical 500-step task.",
                      "896B vector uses the same two-plane writer and a direct latent-vector readout.",
                      "No attempt to equalize propagation FLOPs; parameter counts reported.",
                      "These are 32x32 synthetic 6x6 mark rewrites, not generative image modeling.",
                      "No exact gauge-invariance or holonomy constraint in either learned graph.",
                      "Independent trained seeds, not independent pixel questions."]}
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    # Checkpoint includes weights but no training source pixels.
    torch.save({"seed":seed,"models":{k:m.state_dict() for k,m in models.items()}},
               out/"trained_models.pt")
    print("SEED_RESULT_JSON="+json.dumps({"seed":seed,"evaluation":metrics}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--train-count",type=int,default=128)
    p.add_argument("--test-count",type=int,default=128)
    p.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    p.add_argument("--output",type=Path,default=Path("runs/phase_gate"))
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.steps<1 or args.test_count<4 or args.batch_size<1: p.error("Invalid sizes")
    for seed in args.seeds:
        train_seed(seed,args)


if __name__=="__main__":
    main()
