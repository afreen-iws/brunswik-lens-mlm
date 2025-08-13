
import re, numpy as np, torch
from .encoders import embed_image_clip, embed_text_clip

STOP = set("the a an is are was were on in at to with and or of for this that where who what which how many".split())

def nouns_like(text, k=5):
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    nouns = [w for w in words if w not in STOP and len(w) > 2]
    return nouns[:k]

@torch.no_grad()
def cue_semantic(img, q):
    return torch.cosine_similarity(embed_image_clip(img), embed_text_clip(q)).item()

@torch.no_grad()
def cue_object(img, q):
    ns = nouns_like(q)
    if not ns: return 0.0
    iv = embed_image_clip(img)
    sims = [torch.cosine_similarity(iv, embed_text_clip(f"a photo of a {n}")).item() for n in ns]
    return float(np.mean(sims))

REL_TEMPLATES = [
    ("on",     "a {a} on a {b}",      "a {a} not on a {b}"),
    ("under",  "a {a} under a {b}",   "a {a} not under a {b}"),
    ("nextto", "a {a} next to a {b}", "a {a} far from a {b}"),
]

@torch.no_grad()
def cue_relation(img, q):
    ns = nouns_like(q)
    if len(ns) < 2: return 0.0
    a, b = ns[0], ns[1]
    iv = embed_image_clip(img)
    diffs = []
    for _, pos_t, neg_t in REL_TEMPLATES:
        pos = embed_text_clip(pos_t.format(a=a, b=b))
        neg = embed_text_clip(neg_t.format(a=a, b=b))
        diffs.append(torch.cosine_similarity(iv, pos).item() - torch.cosine_similarity(iv, neg).item())
    return float(np.mean(diffs))
