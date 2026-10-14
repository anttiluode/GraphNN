"""Fixed connectivity, learned gain/phase bytes, fresh complex query activity.

This transport is dissipative. It is not an exact unitary interferometer.
The vector control interprets the same scalar budget without edge transport.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    grid_size: int = 8
    channels: int = 8
    width: int = 32
    steps: int = 4

    def __post_init__(self):
        bounds = {"image_size": (8, 128), "grid_size": (2, 32),
                  "channels": (1, 32), "width": (8, 128), "steps": (2, 16)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        if self.grid_size > self.image_size:
            raise ValueError("grid_size cannot exceed image_size")

    @property
    def payload_bytes(self):
        return 4 * self.channels * self.grid_size * (self.grid_size - 1)


@dataclass(frozen=True)
class GraphMemory:
    # horizontal gain, horizontal phase, vertical gain, vertical phase.
    # Floats are normalized byte values in training; saved parts are uint8.
    parts: tuple[Tensor, Tensor, Tensor, Tensor]

    def flat(self):
        return torch.cat([part.flatten(1) for part in self.parts], dim=1)

    def quantized(self):
        return GraphMemory(tuple(p.detach().clone() if p.dtype == torch.uint8
                                 else (p.detach().clamp(0, 1) * 255).round().to(torch.uint8)
                                 for p in self.parts))

    def to(self, device):
        return GraphMemory(tuple(p.to(device) for p in self.parts))

    def floats(self):
        return tuple(p.float() / 255 if p.dtype == torch.uint8 else p for p in self.parts)

    @classmethod
    def from_flat(cls, flat, config):
        if flat.ndim != 2 or flat.shape[1] != config.payload_bytes:
            raise ValueError("memory has the wrong byte count")
        n = config.channels * config.grid_size * (config.grid_size - 1)
        parts = flat.split(n, dim=1)
        g, c, b = config.grid_size, config.channels, flat.shape[0]
        return cls((parts[0].reshape(b, c, g, g-1), parts[1].reshape(b, c, g, g-1),
                    parts[2].reshape(b, c, g-1, g), parts[3].reshape(b, c, g-1, g)))


def _byte_forward(value):
    rounded = (value.clamp(0, 1) * 255).round() / 255
    return value + (rounded - value).detach()


def _angles(value, batch, device):
    angle = torch.as_tensor(value, dtype=torch.float32, device=device)
    if angle.numel() == 1:
        angle = angle.reshape(1).expand(batch)
    if angle.shape != (batch,) or not bool(torch.isfinite(angle).all()):
        raise ValueError("angle must be finite and scalar or [batch]")
    if bool(((angle < 0) | (angle > 90)).any()):
        raise ValueError("angle must be in 0..90 degrees")
    return angle * (math.pi / 180)


class EdgeWriter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Conv2d(3, cfg.width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(cfg.width, cfg.width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(cfg.width, cfg.width, 3, stride=2, padding=1), nn.SiLU())
        self.proposal = nn.Conv2d(cfg.width + 2, 2 * cfg.channels, 1)
        self.update = nn.Sequential(
            nn.Conv2d(4 * cfg.channels + 2, 2 * cfg.channels, 1), nn.SiLU(),
            nn.Conv2d(2 * cfg.channels, 2 * cfg.channels, 1))

    def forward(self, image, angle, previous):
        f = F.adaptive_avg_pool2d(self.encoder(image), (self.cfg.grid_size, self.cfg.grid_size))
        context = torch.stack((angle.sin(), angle.cos()), dim=1)[..., None, None]
        def edge(a, b, old):
            ctx = context.expand(-1, -1, a.shape[-2], a.shape[-1])
            proposal = torch.sigmoid(self.proposal(torch.cat(((a+b) * .5, ctx), dim=1)))
            gate = torch.sigmoid(self.update(torch.cat((proposal, old, ctx), dim=1)))
            return _byte_forward(old * (1-gate) + proposal * gate)
        old_h = torch.cat(previous[:2], dim=1)
        old_v = torch.cat(previous[2:], dim=1)
        h = edge(f[..., :-1], f[..., 1:], old_h)
        v = edge(f[..., :-1, :], f[..., 1:, :], old_v)
        hg, hp = h.chunk(2, dim=1)
        vg, vp = v.chunk(2, dim=1)
        return GraphMemory((hg, hp, vg, vp))


class JanusModel(nn.Module):
    def __init__(self, config=None, reader="graph"):
        super().__init__()
        self.config = config or ModelConfig()
        if reader not in ("graph", "vector"):
            raise ValueError("reader must be graph or vector")
        self.reader = reader
        c, w = self.config.channels, self.config.width
        self.writer = EdgeWriter(self.config)
        self.ping = nn.Conv2d(6, 2*c, 1)
        self.decoder = nn.Sequential(nn.Conv2d(6*c, w, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(w, w, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(w, 3, 1), nn.Sigmoid())

    @property
    def payload_bytes(self):
        return self.config.payload_bytes

    def _check(self, memory, batch=None):
        if not isinstance(memory, GraphMemory) or len(memory.parts) != 4:
            raise ValueError("expected four memory planes")
        g, c = self.config.grid_size, self.config.channels
        b = memory.parts[0].shape[0] if memory.parts[0].ndim else 0
        shapes = [(b,c,g,g-1)]*2 + [(b,c,g-1,g)]*2
        device = next(self.parameters()).device
        for p, shape in zip(memory.parts, shapes):
            if p.shape != shape or p.device != device:
                raise ValueError("memory shape or device is incompatible")
            if p.dtype != torch.uint8 and (not p.is_floating_point() or
                    not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any())):
                raise ValueError("memory must be uint8 or finite normalized floats")
        if b < 1 or (batch is not None and batch != b):
            raise ValueError("memory batch is incompatible")
        return b

    def write(self, image, context, previous=None):
        size = self.config.image_size
        if image.ndim != 4 or image.shape[1:] != (3,size,size) or image.shape[0] < 1:
            raise ValueError(f"image must be [batch,3,{size},{size}]")
        if not image.is_floating_point() or not bool(torch.isfinite(image).all()) or bool(((image<0)|(image>1)).any()):
            raise ValueError("image must be finite floating RGB in 0..1")
        angle = _angles(context, image.shape[0], image.device)
        if previous is None:
            b, g, c = image.shape[0], self.config.grid_size, self.config.channels
            old = (image.new_full((b,c,g,g-1), .5), image.new_full((b,c,g,g-1), .5),
                   image.new_full((b,c,g-1,g), .5), image.new_full((b,c,g-1,g), .5))
        else:
            self._check(previous, image.shape[0])
            old = previous.floats()
        return self.writer(image, angle, old)

    def encode_pair(self, pair):
        size = self.config.image_size
        if pair.ndim != 5 or pair.shape[1:] != (2,3,size,size):
            raise ValueError("pair must be [batch,2,3,size,size]")
        return self.write(pair[:,1], 90., self.write(pair[:,0], 0.))

    def _fresh_ping(self, angle):
        g = self.config.grid_size
        axis = torch.linspace(-1,1,g,device=angle.device)
        y,x = torch.meshgrid(axis,axis,indexing="ij")
        x,y = x.expand(len(angle),-1,-1), y.expand(len(angle),-1,-1)
        s = angle.sin()[:,None,None].expand_as(x)
        c = angle.cos()[:,None,None].expand_as(x)
        features = torch.stack((x,y,torch.sin(math.pi*x),torch.cos(math.pi*y),s,c),dim=1)
        r,i = torch.tanh(self.ping(features)).chunk(2,dim=1)
        return torch.complex(r,i)

    @staticmethod
    def _propagate(field, parts):
        hg,hp,vg,vp = parts
        h = .05 + .95*hg
        v = .05 + .95*vg
        hc = torch.polar(h, (hp*2-1)*math.pi)
        vc = torch.polar(v, (vp*2-1)*math.pi)
        neighbors = (F.pad(hc*field[...,1:], (0,1,0,0))
                     + F.pad(hc.conj()*field[...,:-1], (1,0,0,0))
                     + F.pad(vc*field[...,1:,:], (0,0,0,1))
                     + F.pad(vc.conj()*field[...,:-1,:], (0,0,1,0)))
        degree = F.pad(h,(0,1,0,0))+F.pad(h,(1,0,0,0))+F.pad(v,(0,0,0,1))+F.pad(v,(0,0,1,0))
        return .5*field + .5*neighbors/degree.clamp_min(1e-6)

    def read(self, memory, angle):
        batch = self._check(memory)
        theta = _angles(angle,batch,memory.parts[0].device)
        field = self._fresh_ping(theta)
        parts = memory.floats()
        if self.reader == "graph":
            responses = [field.real,field.imag]
            for depth in range(1,self.config.steps+1):
                field = self._propagate(field,parts)
                if depth in (self.config.steps//2,self.config.steps):
                    responses.extend((field.real,field.imag))
        else:
            # A flattened scalar code, laid out for the same convolutional decoder.
            # No edge multiplication or graph propagation occurs in this control.
            hg,hp,vg,vp = parts
            responses = [field.real,field.imag,F.pad(hg,(0,1,0,0)),F.pad(hp,(0,1,0,0)),
                         F.pad(vg,(0,0,0,1)),F.pad(vp,(0,0,0,1))]
        features = F.interpolate(torch.cat(responses,dim=1),size=(self.config.image_size,)*2,
                                 mode="bilinear",align_corners=False)
        return self.decoder(features)
