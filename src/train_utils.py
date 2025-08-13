
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_epoch_baseline(model, loader, opt=None):
    ce = nn.CrossEntropyLoss()
    logits_all, y_all, total = [], [], 0.0
    model.train() if opt else model.eval()
    for h, z, y in loader:
        h, y = h.to(device), y.to(device)
        logits = model(h); loss = ce(logits, y)
        if opt: opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * y.size(0)
        logits_all.append(logits.detach().cpu()); y_all.append(y.cpu())
    L = torch.cat(logits_all); Y = torch.cat(y_all)
    acc = accuracy_score(Y.numpy(), L.argmax(1).numpy())
    return total/len(loader.dataset), acc, L, Y

def run_epoch_lens(model, loader, opt=None):
    ce = nn.CrossEntropyLoss()
    logits_all, y_all, total = [], [], 0.0
    model.train() if opt else model.eval()
    for h, z, y in loader:
        h, z, y = h.to(device), z.to(device), y.to(device)
        logits, _ = model(h, z)
        loss = ce(logits, y) + model.lens_regularizer()
        if opt: opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * y.size(0)
        logits_all.append(logits.detach().cpu()); y_all.append(y.cpu())
    L = torch.cat(logits_all); Y = torch.cat(y_all)
    acc = accuracy_score(Y.numpy(), L.argmax(1).numpy())
    return total/len(loader.dataset), acc, L, Y

class TempScaler(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(()))
    def forward(self, logits): return logits / self.T.clamp_min(1e-3)

def fit_temperature(val_logits, val_labels, steps=200):
    scaler = TempScaler().to(val_logits.device)
    opt = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=steps)
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = ce(scaler(val_logits), val_labels); loss.backward(); return loss
    opt.step(closure); return scaler

def ece_score(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1); correct = (pred == labels).float()
    bins = torch.linspace(0,1,n_bins+1); ece = torch.tensor(0.0)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.any(): ece += (correct[m].mean() - conf[m].mean()).abs() * m.float().mean()
    return float(ece)
