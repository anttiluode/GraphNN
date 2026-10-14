"""A → B → revised A: controlled detail recovery and interference measurement."""
from dataclasses import dataclass
import time

import torch

from .model import GraphMemory


PALETTE = ((.12,.40,.95),(.95,.35,.10))


@dataclass(frozen=True)
class RewriteBatch:
    original: torch.Tensor
    updated: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    new_a_labels: torch.Tensor

    def select(self, indices):
        return RewriteBatch(*(getattr(self,name)[indices] for name in
                              ("original","updated","masks","labels","new_a_labels")))

    def to(self, device):
        return RewriteBatch(*(getattr(self,name).to(device) for name in
                              ("original","updated","masks","labels","new_a_labels")))


def make_rewrite_batch(pair, seed=41, controlled_marks=True):
    if pair.ndim!=5 or pair.shape[1:3]!=(2,3) or pair.shape[-1]!=pair.shape[-2] or pair.shape[-1]<8:
        raise ValueError("Expected [batch,2,3,size,size] with square size >=8")
    if not pair.is_floating_point() or not bool(torch.isfinite(pair).all()) or bool(((pair<0)|(pair>1)).any()):
        raise ValueError("Images must be finite floating RGB in 0..1")
    b,s=pair.shape[0],pair.shape[-1]
    side=min(6,max(2,s//4))
    rng=torch.Generator().manual_seed(seed)
    positions=torch.randint(0,s-side+1,(b,2,2),generator=rng)
    labels=torch.randint(0,2,(b,2),generator=rng).to(pair.device)
    palette=pair.new_tensor(PALETTE)
    original=pair.clone()
    masks=pair.new_zeros((b,2,1,s,s))
    for i in range(b):
        for role in range(2):
            y,x=positions[i,role].tolist()
            masks[i,role,:,y:y+side,x:x+side]=1
            if controlled_marks:
                original[i,role,:,y:y+side,x:x+side]=palette[labels[i,role]][:,None,None]
            else:
                mean=original[i,role,:,y:y+side,x:x+side].mean((-2,-1))
                labels[i,role]=((palette-mean).square().sum(1)).argmin()
    updated=original.clone()
    new_labels=1-labels[:,0]
    for i in range(b):
        y,x=positions[i,0].tolist()
        updated[i,0,:,y:y+side,x:x+side]=palette[new_labels[i]][:,None,None]
    return RewriteBatch(original,updated,masks,labels,new_labels)


def _mark_mse(prediction, target, mask):
    return ((prediction-target).square()*mask).sum()/((mask.sum()*3).clamp_min(1))


def fit_rewrite_step(models, optimizers, batch):
    losses={}
    for arm,model in models.items():
        model.train()
        optimizer=optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        a=model.write(batch.original[:,0],0.)
        ab=model.write(batch.original[:,1],90.,a)
        aba=model.write(batch.updated[:,0],0.,ab)
        reads=((a,0,0.,batch.original[:,0]),
               (ab,0,0.,batch.original[:,0]),(ab,1,90.,batch.original[:,1]),
               (aba,0,0.,batch.updated[:,0]),(aba,1,90.,batch.original[:,1]))
        terms=[]
        for memory,role,angle,target in reads:
            prediction=model.read(memory,angle)
            terms.append((prediction-target).square().mean()+.5*_mark_mse(prediction,target,batch.masks[:,role]))
        loss=torch.stack(terms).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Rewrite training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses[arm]=float(loss.detach())
    return losses


@torch.no_grad()
def rewrite_states(model, batch):
    a=model.write(batch.original[:,0],0.).quantized()
    ab=model.write(batch.original[:,1],90.,a).quantized()
    # The update sees A' and the previous bytes; no B source pixels are replayed.
    aba=model.write(batch.updated[:,0],0.,ab).quantized()
    return ab,aba


@torch.no_grad()
def evaluate_rewrites(models, batch):
    if len(batch.original)<2:
        raise ValueError("At least two examples are needed for memory shuffle")
    results={}
    for arm,model in models.items():
        model.eval()
        start=time.perf_counter()
        before,after=rewrite_states(model,batch)
        palette=batch.original.new_tensor(PALETTE)
        def score(memory,new):
            outputs=[model.read(memory,angle) for angle in (0.,90.)]
            targets=batch.updated if new else batch.original
            labels=(batch.new_a_labels if new else batch.labels[:,0],batch.labels[:,1])
            correct=[]
            report={}
            for role,name in enumerate(("a","b")):
                mask=batch.masks[:,role]
                color=(outputs[role]*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)
                predicted=(color[:,None,:]-palette[None,:,:]).square().sum(-1).argmin(-1)
                hit=predicted==labels[role]
                correct.append(hit)
                report[name+"_mark_accuracy"]=float(hit.float().mean())
                report[name+"_rgb_mse"]=float((outputs[role]-targets[:,role]).square().mean())
                report[name+"_mark_mse"]=float(_mark_mse(outputs[role],targets[:,role],mask))
            outside=1-batch.masks[:,0]
            report["a_outside_mse"]=float(_mark_mse(outputs[0],targets[:,0],outside))
            report["joint_mark_accuracy"]=float((correct[0]&correct[1]).float().mean())
            return report,outputs
        before_score,before_rgb=score(before,False)
        after_score,after_rgb=score(after,True)
        shuffled=GraphMemory(tuple(p.roll(1,0) for p in after.parts))
        rebuilt_a=model.write(batch.updated[:,0],0.).quantized()
        rebuilt=model.write(batch.updated[:,1],90.,rebuilt_a).quantized()
        results[arm]={"before":before_score,"after":after_score,
                      "no_write":score(before,True)[0],"shuffled":score(shuffled,True)[0],
                      "rebuilt":score(rebuilt,True)[0],
                      "b_prediction_change_mse":float((after_rgb[1]-before_rgb[1]).square().mean()),
                      "changed_byte_fraction":float((before.flat()!=after.flat()).float().mean()),
                      "payload_bytes":model.payload_bytes,
                      "shared_parameters":sum(p.numel() for p in model.parameters()),
                      "shared_weight_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
                      "evaluation_seconds":time.perf_counter()-start}
    return results
