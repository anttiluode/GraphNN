"""Inspect 896-byte phase/gain/vector and PR2 models side by side in a real GUI.

After training/download of GitHub Actions artifact:
    python phase_computer_gui.py --seed 41 --runs runs/phase_gate

This viewer displays only held-out procedural examples, not training images.
It never trains weights. The new models are genuinely 896-byte memories.
"""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from PIL import Image, ImageTk
from graphnn.data import procedural_pairs
from graphnn.rewrite import make_rewrite_batch
from graphnn.half_memory import HalfJanus
from graphnn.training import PairSession

ALIASES=(("phase_only_896","Phase-only 896B"),
         ("gain_only_896","Gain-only 896B"),
         ("vector_896","Vector 896B"),
         ("graph_1792_original","Graph 1792B"),
         ("vector_1792_original","Vector 1792B"))


class PhaseComputerLab(tk.Tk):
    def __init__(self,seed=41,runs="runs/phase_gate"):
        super().__init__()
        torch.set_num_threads(2)
        self.title("GraphNN · phase computer laboratory")
        self.geometry("1190x570")
        self.seed=seed
        self.runs=Path(runs)
        self.models={}
        self.images={}
        self.pictures=[]
        self.sample=tk.IntVar(value=0)
        self.angle=tk.DoubleVar(value=90.0)
        self.updated=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value="Loading frozen models...")
        opts=ttk.Frame(self,padding=9)
        opts.pack(fill="x")
        ttk.Label(opts,text="Test example").pack(side="left")
        ttk.Spinbox(opts,from_=0,to=127,textvariable=self.sample,
                    command=self.render,width=5).pack(side="left",padx=5)
        ttk.Label(opts,text="Query angle").pack(side="left",padx=(16,3))
        ttk.Scale(opts,from_=0,to=90,variable=self.angle,
                  length=215,command=lambda _:self.render()).pack(side="left")
        ttk.Checkbutton(opts,text="After A' rewrite",variable=self.updated,
                        command=self.render).pack(side="left",padx=14)
        ttk.Button(opts,text="Render",command=self.render).pack(side="left")
        ttk.Button(opts,text="Next example",command=self.next_sample).pack(side="left",padx=6)
        ttk.Label(self,textvariable=self.status,wraplength=1110,
                  justify="left").pack(anchor="w",padx=10,pady=(0,9))
        canvas=ttk.Frame(self)
        canvas.pack(fill="both",expand=True)
        self.tiles=[]
        for i,label in enumerate(("Target","Phase-only 896B","Gain-only 896B",
                                  "Direct vector 896B","Complex graph 1792B","Vector 1792B")):
            col=ttk.Frame(canvas,padding=4)
            col.grid(row=0,column=i,sticky="n")
            ttk.Label(col,text=label).pack()
            photo=ttk.Label(col)
            photo.pack(pady=6)
            desc=ttk.Label(col,text="",wraplength=175,justify="center")
            desc.pack()
            self.tiles.append((photo,desc))
        self.help=ttk.Label(self,text="0° queries A; 90° queries B. Intermediate angles are unsupervised interpolations, not physical views. The 896-byte checkpoints must first be trained or downloaded from the workflow artifact.",wraplength=1120)
        self.help.pack(fill="x",padx=9,pady=7)
        self.load_models()
        self.render()

    def next_sample(self):
        self.sample.set((self.sample.get()+1)%128)
        self.render()

    def load_models(self):
        check=self.runs/f"seed{self.seed}"/"trained_models.pt"
        if not check.exists():
            raise FileNotFoundError(
                f"Missing {check}. Train with: python train_phase_gate.py "
                f"--steps 500 --seeds {self.seed} --device cpu --output {self.runs} "
                f"or download the GitHub Actions phase-gate-seed-{self.seed} artifact.")
        saved=torch.load(check,map_location="cpu",weights_only=True)
        if saved.get("seed")!=self.seed: raise ValueError("wrong checkpoint seed")
        for k in ("phase_only_896","gain_only_896","vector_896"):
            kind="gain_only" if k=="gain_only_896" else "phase_only"
            reader="vector" if k=="vector_896" else "graph"
            torch.manual_seed(self.seed)
            m=HalfJanus(kind=kind,reader=reader)
            m.load_state_dict(saved["models"][k])
            self.models[k]=m.eval()
        reference=Path(__file__).resolve().parent/"results/rewrite_gate/pilot500"/f"seed{self.seed}"/"memory.pt"
        control=PairSession(device="cpu",seed=self.seed)
        control.load(reference)
        self.models["graph_1792_original"]=control.models["graph"].eval()
        self.models["vector_1792_original"]=control.models["vector"].eval()
        self.data=make_rewrite_batch(procedural_pairs(128,seed=self.seed+10000),
                                     seed=self.seed+20000)
        self.status.set(f"Frozen test models loaded • seed {self.seed} • both variants trained 500 steps • 896B persistent half-memory")

    def render(self):
        if not self.models:return
        try:
            index=int(self.sample.get())
            if index<0 or index>=128:return
            angle=float(self.angle.get())
            is_after=bool(self.updated.get())
            targets=self.data.updated if is_after else self.data.original
            target=(targets[index,0 if angle<45 else 1].permute(1,2,0).numpy())
            output=[(target,"held-out target" if angle in (0.,90.) else "nearest endpoint target (not an interpolation)")]
            self.pictures=[]
            with torch.no_grad():
                for name,_ in ALIASES:
                    m=self.models[name]
                    a=self.data.original[index:index+1,0]
                    b=self.data.original[index:index+1,1]
                    a2=self.data.updated[index:index+1,0]
                    first=m.write(a,0.).quantized()
                    prior=m.write(b,90.,first).quantized()
                    after=m.write(a2,0.,prior).quantized()
                    state=after if is_after else prior
                    pred=m.read(state,angle)[0].permute(1,2,0).numpy()
                    old_b=m.read(prior,90.)
                    new_b=m.read(after,90.)
                    drift=float((old_b-new_b).square().mean())
                    bytes_=m.payload_bytes
                    output.append((pred,f"{bytes_} stored bytes • B drift {drift:.5f}"))
            for (array,caption),(tile,desc) in zip(output,self.tiles):
                image=Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8))
                bitmap=ImageTk.PhotoImage(image.resize((176,176),Image.Resampling.NEAREST))
                tile.configure(image=bitmap)
                tile.image=bitmap
                desc.configure(text=caption)
                self.pictures.append(bitmap)
        except Exception as e:
            self.status.set("Viewer error: "+str(e))


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--seed",type=int,choices=(41,97,127),default=41)
    a.add_argument("--runs",default="runs/phase_gate")
    opts=a.parse_args()
    try:
        PhaseComputerLab(seed=opts.seed,runs=opts.runs).mainloop()
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        raise SystemExit(2) from e


if __name__=="__main__":
    main()
