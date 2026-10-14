"""Genuine 896-byte phase-only/gain-only graph memories.

No dormant 896-byte second memory: precisely two byte planes (horizontal and
vertical) persist between calls. Four-plane GraphMemory is only a TEMPORARY
reader view, with the missing gain/phase planes filled by fixed constants.
The graph reader uses exactly GraphNN's original dissipative complex transport.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import JanusModel, GraphMemory, ModelConfig, _angles, _byte_forward


@dataclass(frozen=True)
class HalfMemory:
    parts: tuple[torch.Tensor, torch.Tensor]

    def flat(self):
        return torch.cat([p.flatten(1) for p in self.parts], dim=1)

    def floats(self):
        return tuple(p.float() / 255.0 if p.dtype == torch.uint8 else p for p in self.parts)

    def quantized(self):
        return HalfMemory(tuple(p.clone().detach() if p.dtype == torch.uint8 else
                                (p.detach().clamp(0, 1)*255).round().to(torch.uint8)
                                for p in self.parts))


class HalfEdgeWriter(nn.Module):
    """Single gain OR phase byte per channel/edge, same architecture for both."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Conv2d(3, cfg.width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(cfg.width, cfg.width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(cfg.width, cfg.width, 3, stride=2, padding=1), nn.SiLU())
        self.proposal = nn.Conv2d(cfg.width + 2, cfg.channels, 1)
        self.update = nn.Sequential(
            nn.Conv2d(2*cfg.channels+2, cfg.channels, 1), nn.SiLU(),
            nn.Conv2d(cfg.channels, cfg.channels, 1))

    def forward(self, image, angle, old):
        features = F.adaptive_avg_pool2d(self.encoder(image),
                                          (self.cfg.grid_size,self.cfg.grid_size))
        context = torch.stack((angle.sin(),angle.cos()), dim=1)[...,None,None]
        def one(left,right,prior):
            ctx=context.expand(-1,-1,left.shape[-2],left.shape[-1])
            proposal=torch.sigmoid(self.proposal(torch.cat(((left+right)*0.5,ctx),dim=1)))
            gate=torch.sigmoid(self.update(torch.cat((proposal,prior,ctx),dim=1)))
            return _byte_forward(prior*(1-gate)+proposal*gate)
        horizontal=one(features[...,:,:-1], features[...,1:], old[0])
        vertical=one(features[...,:-1,:], features[...,1:,:], old[1])
        return HalfMemory((horizontal,vertical))


class HalfJanus(nn.Module):
    """One persistent byte per edge-channel; fixed other factor.

    kind is phase_only or gain_only. reader is graph or vector.
    phase_only (graph) fixes all edge gains to a constant 0.5;
    gain_only (graph) fixes all edge phases to neutral midpoint 0.5.
    A vector reader sees two byte planes as flattened features and uses
    precisely the same parameter budget as its graph counterpart.
    """
    def __init__(self, config=None, kind="phase_only", reader="graph"):
        super().__init__()
        if kind not in ("phase_only","gain_only"):
            raise ValueError("kind")
        if reader not in ("graph","vector"):
            raise ValueError("reader")
        self.config=config or ModelConfig()
        self.kind=kind
        self.reader=reader
        self.writer=HalfEdgeWriter(self.config)
        # Shared readout follows original JanusModel shapes and transport.
        self.backbone=JanusModel(self.config,reader=reader)
        # The backbone's image writer is unused; do not keep/count it.
        self.backbone.writer=nn.Identity()
        self.payload_bytes=2*self.config.channels*self.config.grid_size*(self.config.grid_size-1)

    def write(self,image,context,previous=None):
        size=self.config.image_size
        if image.ndim!=4 or image.shape[1:]!=(3,size,size):
            raise ValueError("write expects [B,3,S,S]")
        angle=_angles(context,image.size(0),image.device)
        c,g=self.config.channels,self.config.grid_size
        if previous is None:
            old=(image.new_full((len(image),c,g,g-1),.5),
                 image.new_full((len(image),c,g-1,g),.5))
        else:
            if not isinstance(previous,HalfMemory):
                raise ValueError("previous must be HalfMemory")
            old=previous.floats()
        return self.writer(image,angle,old)

    def encode_pair(self,pair):
        return self.write(pair[:,1],90.,self.write(pair[:,0],0.).quantized())

    def expanded(self,m):
        if not isinstance(m,HalfMemory):
            raise ValueError("expected HalfMemory")
        h,v=m.floats()
        # For the direct baseline, supply the *same* two planes in the
        # same ordered slot locations for both phase and gain variants.
        # The direct read does not physically require phase semantics.
        const_h=torch.full_like(h,0.5)
        const_v=torch.full_like(v,0.5)
        if self.reader=="vector":
            return GraphMemory((h,const_h,v,const_v))
        if self.kind=="phase_only":
            return GraphMemory((const_h,h,const_v,v))
        return GraphMemory((h,const_h,v,const_v))

    def read(self,m,angle):
        return self.backbone.read(self.expanded(m),angle)
