"""Small deterministic procedural pairs; no external data or image service."""
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch


def demo_pair(size=32, seed=41):
    if not isinstance(size,int) or not 8 <= size <= 128:
        raise ValueError("size must be in 8..128")
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(2):
        base = rng.integers(8,45,3)
        image = Image.new("RGB",(size,size),tuple(int(v) for v in base))
        draw = ImageDraw.Draw(image)
        for j in range(4):
            x,y = rng.integers(1,max(2,size//2),2)
            w,h = rng.integers(max(2,size//5),max(3,size//2),2)
            box = (int(x),int(y),min(size-2,int(x+w)),min(size-2,int(y+h)))
            color = tuple(int(v) for v in rng.integers(70,250,3))
            if j % 2:
                draw.ellipse(box,fill=color)
            else:
                draw.rectangle(box,fill=color)
        # A small asymmetric identity detail, relevant to retention at this scale.
        x,y = rng.integers(1,size-2,2)
        draw.rectangle((int(x),int(y),int(x+1),int(y+1)),fill=(250,235,50))
        images.append(np.asarray(image,dtype=np.float32)/255.)
    return np.stack(images)


def procedural_pairs(count, size=32, seed=41):
    if type(count) is not int or count < 2:
        raise ValueError("count must be at least two")
    pairs = np.stack([demo_pair(size,seed*100003+i) for i in range(count)])
    return torch.from_numpy(pairs.transpose(0,1,4,2,3).copy())


def load_image(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize((size,size),Image.Resampling.LANCZOS)
        return np.asarray(image,dtype=np.float32)/255.


def image_from_array(array):
    return Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8))
