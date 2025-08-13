
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

def load_cifar_vqa(N_TRAIN=5000, N_VAL=1000, N_TEST=1000, seed=42):
    cifar = load_dataset("cifar10")
    names = cifar["train"].features["label"].names

    train_raw = cifar["train"].shuffle(seed=seed).select(range(N_TRAIN + N_VAL))
    test_raw  = cifar["test"].shuffle(seed=seed).select(range(N_TEST))

    # wrap into VQA triples: (image, question, answer)
    def map_row(r):
        img = r["img"] if "img" in r else r["image"]
        return {"image": img, "question": "What object is in the image?", "answer": names[int(r["label"])]}
    train_small = train_raw.map(map_row)
    val_small   = train_raw.select(range(N_TRAIN, N_TRAIN+N_VAL)).map(map_row)
    train_small = train_raw.select(range(N_TRAIN)).map(map_row)
    test_small  = test_raw.map(map_row)
    return {"train": train_small, "val": val_small, "test": test_small}

def build_vocab(ds_split, K=10):
    vocab = sorted(list({str(rec["answer"]) for rec in ds_split}))[:K]
    a2id = {a:i for i,a in enumerate(vocab)}
    return a2id, vocab

from src.encoders import embed_image_clip, embed_text_distilbert
from ..cues import cue_semantic, cue_object, cue_relation

class VQACIFARDataset(Dataset):
    def __init__(self, hf_split, a2id):
        self.rows = [r for r in hf_split if str(r["answer"]) in a2id]
        self.a2id = a2id
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = r["image"]; q = str(r["question"]); a = str(r["answer"])
        h = torch.cat([embed_image_clip(img), embed_text_distilbert(q)], dim=1).squeeze(0)
        z = torch.tensor([cue_semantic(img, q), cue_object(img, q), cue_relation(img, q)], dtype=torch.float32)
        y = torch.tensor(self.a2id[a], dtype=torch.long)
        return h.cpu(), z.cpu(), y

def make_loaders(N_TRAIN=5000, N_VAL=1000, N_TEST=1000, K=10, batch=32, seed=42):
    splits = load_cifar_vqa(N_TRAIN, N_VAL, N_TEST, seed)
    a2id, vocab = build_vocab(splits["train"], K=K)
    train_ds = VQACIFARDataset(splits["train"], a2id)
    val_ds   = VQACIFARDataset(splits["val"],   a2id)
    test_ds  = VQACIFARDataset(splits["test"],  a2id)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=0)
    sample_h, sample_z, sample_y = next(iter(train_loader))
    h_dim = sample_h.shape[1]
    return train_loader, val_loader, test_loader, h_dim, len(a2id), vocab
