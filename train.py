"""Train reusable writer/readers on independent procedural pairs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from graphnn.data import procedural_pairs, image_from_array
from graphnn.model import ModelConfig
from graphnn.training import PairSession, evaluate, fit_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--train-count",type=int,default=128)
    p.add_argument("--test-count",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--output",type=Path,default=Path("runs/pilot"))
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    torch.set_num_threads(2)
    session = PairSession(device=args.device,seed=args.seed)
    training = procedural_pairs(args.train_count,seed=args.seed)
    test = procedural_pairs(args.test_count,seed=args.seed+10000)
    rng = torch.Generator().manual_seed(args.seed)
    start = time.perf_counter()
    history = []
    for step in range(1,args.steps+1):
        idx = torch.randint(0,len(training),(args.batch_size,),generator=rng)
        losses = fit_step(session.models,session.optimizers,training[idx].to(session.device))
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append({"step":step,**losses})
            print(f"{step}/{args.steps}: graph={losses['graph']:.5f} vector={losses['vector']:.5f}",flush=True)
    results = {"seed":args.seed,"steps":args.steps,"config":asdict(session.config),
               "train_count":len(training),"test_count":len(test),"batch_size":args.batch_size,
               "device":str(session.device),"torch_version":str(torch.__version__),
               "training_seconds":time.perf_counter()-start,"history":history,
               "held_out":evaluate(session.models,test),
               "limits":"One seed, procedural 32x32 pairs; equal bytes and parameters, different inference computation. No natural-photo or unlimited-capacity claim."}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"metrics.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    session.steps,session.mode = args.steps,"reusable-procedural"
    session.set_pair(test[0].permute(0,2,3,1).numpy())
    session.save(args.output/"memory.pt")
    tile = 128
    sheet = Image.new("RGB",(tile*4,tile*4+28),(15,20,30))
    draw = ImageDraw.Draw(sheet)
    for i,label in enumerate(("Target A/B","Graph","Vector","Graph randomized")):
        draw.text((i*tile+8,8),label,fill="white")
    for row in range(4):
        session.set_pair(test[row//2].permute(0,2,3,1).numpy())
        angle = 90.*(row%2)
        answers = session.render(angle)
        arrays = [session.images[row%2],answers["graph"],answers["vector"],session.render(angle,"randomized")["graph"]]
        for col,array in enumerate(arrays):
            sheet.paste(image_from_array(array).resize((tile,tile),Image.Resampling.NEAREST),(col*tile,row*tile+28))
    sheet.save(args.output/"predictions.png")
    print(json.dumps(results["held_out"],indent=2))


if __name__ == "__main__":
    main()
