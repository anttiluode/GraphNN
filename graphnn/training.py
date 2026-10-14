"""Independent matched training, memory-only inference, portable sessions."""
from dataclasses import asdict
import math
import os
from pathlib import Path
from threading import RLock
import tempfile
import time

import numpy as np
import torch

from .model import GraphMemory, JanusModel, ModelConfig


def choose_device(requested="auto"):
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    if device not in ("cpu","cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Use cpu, auto, or cuda with a working CUDA PyTorch installation")
    return torch.device(device)


def fit_step(models, optimizers, pair):
    losses = {}
    for arm,model in models.items():
        model.train()
        optimizer = optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        memory = model.encode_pair(pair)
        loss = sum((model.read(memory,angle)-pair[:,i]).square().mean()
                   for i,angle in enumerate((0.,90.))) / 2
        if not bool(torch.isfinite(loss)):
            raise ValueError("Training became non-finite; reset and use a smaller learning rate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses[arm] = float(loss.detach())
    return losses


@torch.no_grad()
def evaluate(models, pairs):
    if pairs.shape[0] < 2:
        raise ValueError("Shuffle control requires at least two independent pairs")
    results = {}
    for arm,model in models.items():
        model.eval()
        pair = pairs.to(next(model.parameters()).device)
        start = time.perf_counter()
        memory = model.encode_pair(pair).quantized()
        def score(state):
            return sum(float((model.read(state,a)-pair[:,i]).square().mean())
                       for i,a in enumerate((0.,90.))) / 2
        mse = score(memory)
        shuffled = GraphMemory(tuple(p.roll(1,0) for p in memory.parts))
        blank = GraphMemory(tuple(torch.full_like(p,128) for p in memory.parts))
        results[arm] = {"mse":mse,"psnr_db":-10*math.log10(max(mse,1e-12)),
                        "shuffled_mse":score(shuffled),"blank_mse":score(blank),
                        "payload_bytes":model.payload_bytes,
                        "shared_parameters":sum(p.numel() for p in model.parameters()),
                        "shared_weight_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
                        "evaluation_seconds":time.perf_counter()-start}
    return results


class PairSession:
    def __init__(self, config=None, device="auto", seed=41, learning_rate=.003):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.lock = RLock()
        self.config = config or ModelConfig()
        self.device = choose_device(device)
        self.seed = seed
        self.learning_rate = learning_rate
        self.models = {}
        for arm in ("graph","vector"):
            torch.manual_seed(seed)
            self.models[arm] = JanusModel(self.config,arm).to(self.device)
        self.optimizers = {k:torch.optim.Adam(m.parameters(),lr=learning_rate) for k,m in self.models.items()}
        self.images = None
        self.memories = {}
        self.steps = 0
        self.mode = "untrained"

    def set_pair(self, images):
        images = np.asarray(images,dtype=np.float32)
        s = self.config.image_size
        if images.shape != (2,s,s,3) or not np.isfinite(images).all() or ((images<0)|(images>1)).any():
            raise ValueError(f"Expected two finite {s}x{s} RGB images in 0..1")
        with self.lock:
            self.images = images.copy()
            self.encode()

    def _pair(self):
        if self.images is None:
            raise ValueError("Load both source images before training or encoding")
        return torch.from_numpy(self.images.transpose(0,3,1,2).copy()).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode(self):
        with self.lock:
            pair = self._pair()
            self.memories = {k:m.encode_pair(pair).quantized() for k,m in self.models.items()}

    def step(self):
        with self.lock:
            loss = fit_step(self.models,self.optimizers,self._pair())
            self.steps += 1
            self.mode = "pair-fit"
            self.encode()
            return loss

    def _controlled(self, memory, control):
        if control == "stored":
            return memory
        if control == "blank":
            return GraphMemory(tuple(torch.full_like(p,128) for p in memory.parts))
        if control == "randomized":
            generator = torch.Generator(device="cpu").manual_seed(123)
            return GraphMemory(tuple(torch.randint(0,256,p.shape,generator=generator,dtype=torch.uint8).to(p.device)
                                     for p in memory.parts))
        raise ValueError("Unknown memory control")

    @torch.no_grad()
    def render(self, angle=0., control="stored"):
        with self.lock:
            if not self.memories:
                raise ValueError("No stored memory; load two images or a saved memory")
            return {k:m.read(self._controlled(self.memories[k],control),angle)[0].permute(1,2,0).cpu().numpy()
                    for k,m in self.models.items()}

    @torch.no_grad()
    def metrics(self):
        with self.lock:
            results = {}
            pair = self._pair() if self.images is not None else None
            for arm,model in self.models.items():
                metric = {"payload_bytes":model.payload_bytes,
                          "shared_parameters":sum(p.numel() for p in model.parameters())}
                if pair is not None and arm in self.memories:
                    mse = sum(float((model.read(self.memories[arm],a)-pair[:,i]).square().mean())
                              for i,a in enumerate((0.,90.))) / 2
                    metric.update(mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)))
                results[arm] = metric
            return results

    def maps(self, channel=0):
        with self.lock:
            if not 0 <= channel < self.config.channels:
                raise ValueError("channel outside stored memory")
            parts = self.memories["graph"].parts
            h,p,v,q = (t[0,channel].float().cpu().numpy()/255 for t in parts)
            g = self.config.grid_size
            gain,phase,count = np.zeros((g,g)),np.zeros((g,g)),np.zeros((g,g))
            for values,out in ((h,gain),(p,phase)):
                out[:,:-1] += values
                out[:,1:] += values
            for values,out in ((v,gain),(q,phase)):
                out[:-1,:] += values
                out[1:,:] += values
            count[:,:-1] += 1; count[:,1:] += 1
            count[:-1,:] += 1; count[1:,:] += 1
            return gain/count,phase/count

    def save(self, path):
        with self.lock:
            if not self.memories:
                raise ValueError("There is no memory to save")
            payload = {"format":"graphnn-memory","version":1,"config":asdict(self.config),
                       "models":{k:{n:t.detach().cpu().clone() for n,t in m.state_dict().items()} for k,m in self.models.items()},
                       "memories":{k:v.flat().detach().cpu().clone() for k,v in self.memories.items()},
                       "steps":self.steps,"mode":self.mode}
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary = tempfile.mkstemp(prefix=".graphnn-",suffix=".pt",dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload,temporary)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path):
        # Stage everything before replacing a working session.
        try:
            payload = torch.load(path,map_location="cpu",weights_only=True)
            if not isinstance(payload,dict) or payload.get("format") != "graphnn-memory" or payload.get("version") != 1:
                raise ValueError("This is not a GraphNN memory file")
            config = ModelConfig(**payload["config"])
            models,memories = {},{}
            for arm in ("graph","vector"):
                model = JanusModel(config,arm)
                weights = payload["models"][arm]
                if not all(isinstance(t,torch.Tensor) and bool(torch.isfinite(t).all()) for t in weights.values()):
                    raise ValueError("Non-finite model weights")
                model.load_state_dict(weights,strict=True)
                model = model.to(self.device)
                if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
                    raise ValueError("Model weights overflowed their inference dtype")
                flat = payload["memories"][arm]
                if not isinstance(flat,torch.Tensor) or flat.dtype != torch.uint8 or flat.shape != (1,config.payload_bytes):
                    raise ValueError("Invalid stored byte memory")
                memory = GraphMemory.from_flat(flat.to(self.device),config)
                for angle in (0.,90.):
                    if not bool(torch.isfinite(model.read(memory,angle)).all()):
                        raise ValueError("Saved memory produces a non-finite answer")
                models[arm],memories[arm] = model,memory
            steps,mode = payload.get("steps",0),payload.get("mode","loaded")
            if type(steps) is not int or steps < 0 or not isinstance(mode,str):
                raise ValueError("Invalid training metadata")
        except Exception as error:
            raise ValueError(f"Cannot load GraphNN memory: {error}") from error
        with self.lock:
            self.config,self.models,self.memories = config,models,memories
            self.optimizers = {k:torch.optim.Adam(m.parameters(),lr=self.learning_rate) for k,m in models.items()}
            self.images = None
            self.steps,self.mode = steps,mode
