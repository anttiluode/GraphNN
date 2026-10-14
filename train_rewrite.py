"""Train and measure the A → B → revised A retention gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch

from graphnn.data import procedural_pairs, image_from_array
from graphnn.training import PairSession
from graphnn.rewrite import make_rewrite_batch, fit_rewrite_step, evaluate_rewrites, rewrite_states


def contact_sheet(session, batch, path):
    tile=112
    titles=("A before","A updated","Graph A'","Vector A'","B retained","Graph B","Vector B")
    sheet=Image.new("RGB",(tile*len(titles),tile*4+28),(15,20,30))
    draw=ImageDraw.Draw(sheet)
    for col,title in enumerate(titles): draw.text((col*tile+5,8),title,fill="white")
    for row in range(4):
        example=batch.select(torch.tensor([row],device=batch.original.device))
        answers={}
        for arm,model in session.models.items():
            _,state=rewrite_states(model,example)
            with torch.no_grad():
                answers[arm]=[model.read(state,a)[0].permute(1,2,0).cpu().numpy() for a in (0.,90.)]
        sources=[example.original[0,0],example.updated[0,0],example.original[0,1]]
        sources=[s.permute(1,2,0).cpu().numpy() for s in sources]
        arrays=[sources[0],sources[1],answers["graph"][0],answers["vector"][0],sources[2],answers["graph"][1],answers["vector"][1]]
        for col,array in enumerate(arrays):
            sheet.paste(image_from_array(array).resize((tile,tile),Image.Resampling.NEAREST),(col*tile,row*tile+28))
    sheet.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--train-count",type=int,default=128)
    parser.add_argument("--test-count",type=int,default=128)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--output",type=Path,default=Path("runs/rewrite_gate"))
    args=parser.parse_args()
    if args.steps<1 or args.batch_size<1 or args.train_count<4 or args.test_count<4:
        parser.error("positive steps/batch size and at least four training/test pairs are required")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for seed in args.seeds:
        session=PairSession(device=args.device,seed=seed)
        training=make_rewrite_batch(procedural_pairs(args.train_count,seed=seed),seed=seed+1000)
        test=make_rewrite_batch(procedural_pairs(args.test_count,seed=seed+10000),seed=seed+20000).to(session.device)
        generator=torch.Generator().manual_seed(seed)
        history=[]
        start=time.perf_counter()
        for step in range(1,args.steps+1):
            idx=torch.randint(0,len(training.original),(args.batch_size,),generator=generator)
            loss=fit_rewrite_step(session.models,session.optimizers,training.select(idx).to(session.device))
            if step==1 or step%100==0 or step==args.steps:
                history.append({"step":step,**loss})
                print(f"seed {seed} {step}/{args.steps}: graph={loss['graph']:.5f} vector={loss['vector']:.5f}",flush=True)
        metrics=evaluate_rewrites(session.models,test)
        receipt={"seed":seed,"steps":args.steps,"batch_size":args.batch_size,
                 "train_count":args.train_count,"test_count":args.test_count,"config":asdict(session.config),
                 "device":str(session.device),"torch_version":str(torch.__version__),
                 "training_seconds":time.perf_counter()-start,"history":history,"held_out":metrics,
                 "limits":"A->B->A' only, procedural 32x32 data, equal byte/parameter budgets but different compute. No-write keeps A->B state; shuffled rolls final memories; rebuild replays B pixels and is privileged."}
        legacy=Path(__file__).parent/"results"/"pilot"/"memory.pt"
        if legacy.exists():
            original=PairSession(device=str(session.device));original.load(legacy)
            receipt["original_pair_trained_checkpoint"]=evaluate_rewrites(original.models,test)
        out=args.output/f"seed{seed}"
        out.mkdir(parents=True,exist_ok=True)
        (out/"metrics.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        session.steps,session.mode=args.steps,"reusable-rewrite"
        example=test.select(torch.tensor([0],device=test.original.device))
        session.memories={arm:rewrite_states(model,example)[1] for arm,model in session.models.items()}
        session.save(out/"memory.pt")
        contact_sheet(session,test,out/"predictions.png")
        receipts.append(receipt)
        print(json.dumps({"seed":seed,"after":{arm:metrics[arm]["after"] for arm in metrics}}),flush=True)
    means={arm:{name:sum(r["held_out"][arm]["after"][name] for r in receipts)/len(receipts)
                 for name in ("a_mark_accuracy","b_mark_accuracy","joint_mark_accuracy")} for arm in ("graph","vector")}
    summary={"runs":[{"seed":r["seed"],"metrics":f"seed{r['seed']}/metrics.json"} for r in receipts],
             "mean_after":means,"mechanism_pass":{arm:m["a_mark_accuracy"]>=.90 and m["b_mark_accuracy"]>=.90 and m["joint_mark_accuracy"]>=.80 for arm,m in means.items()},
             "graph_joint_advantage_every_seed":all(r["held_out"]["graph"]["after"]["joint_mark_accuracy"]>r["held_out"]["vector"]["after"]["joint_mark_accuracy"] for r in receipts),
             "criteria":"Mean new A >=90%, retained B >=90%, joint >=80%; graph advantage requires higher joint accuracy in every seed. Exploratory synthetic gate, not a universal architecture claim."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
