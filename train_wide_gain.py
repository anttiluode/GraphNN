"""Additional 1792-byte gain-only graph control (16 edge channels, no phases).

The original full graph uses 8 channels x 2 factors x 112 edges.
This baseline uses 16 gain-only channels x 112 edges = 1792 uint8 bytes.
It has MORE trainable parameters/compute than the 896B phase/gain arms.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from graphnn.model import ModelConfig
from graphnn.half_memory import HalfJanus
from graphnn.data import procedural_pairs
from graphnn.rewrite import make_rewrite_batch, fit_rewrite_step
from train_phase_gate import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",nargs="+",type=int,default=[41,97,127])
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    p.add_argument("--output",type=Path,default=Path("runs/wide_gain_gate"))
    args=p.parse_args()
    torch.set_num_threads(2)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model=HalfJanus(ModelConfig(channels=16),kind="gain_only",reader="graph").to(args.device)
        assert model.payload_bytes==1792
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        train=make_rewrite_batch(procedural_pairs(128,seed=seed),seed=seed+1000)
        test=make_rewrite_batch(procedural_pairs(128,seed=seed+10000),seed=seed+20000)
        generator=torch.Generator().manual_seed(seed)
        start=time.perf_counter()
        history=[]
        for step in range(1,args.steps+1):
            ix=torch.randint(0,128,(8,),generator=generator)
            loss=fit_rewrite_step({"wide_gain":model},{"wide_gain":opt},
                                  train.select(ix).to(args.device))["wide_gain"]
            if step==1 or step%100==0 or step==args.steps:
                history.append({"step":step,"loss":loss})
                print(json.dumps({"seed":seed,"step":step,"loss":loss}),flush=True)
        m=evaluate(model,test.to(args.device))
        record={"seed":seed,"steps":args.steps,"variant":"gain_only_1792_16channels",
                "metrics":m,"training":history,"seconds":time.perf_counter()-start,
                "caveats":"16 channels and more shared parameters than phase-only 8-channel graph. This is a larger-capacity, more-compute gain-only baseline, not a matched-parameter control."}
        dest=args.output/f"seed{seed}"
        dest.mkdir(parents=True,exist_ok=True)
        (dest/"metrics.json").write_text(json.dumps(record,indent=2)+"\n")
        torch.save(model.state_dict(),dest/"gain_only_1792.pt")
        print("WIDE_RESULT_JSON="+json.dumps(record),flush=True)


if __name__=="__main__":
    main()
