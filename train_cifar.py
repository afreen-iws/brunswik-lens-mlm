
import argparse, numpy as np, torch
from src.data.cifar_vqa import make_loaders
from src.models.lens_head import BaselineHead, LensHead
from src.train_utils import run_epoch_baseline, run_epoch_lens, fit_temperature, ece_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def estimate_cue_validity(loader, num_classes):
    Z, Y = [], []
    for h, z, y in loader:
        Z.append(z.numpy()); Y.append(y.numpy())
    Z = np.concatenate(Z, 0); Y = np.concatenate(Y, 0)
    vals = []
    for k in range(3):
        overall = np.var(Z[:,k]) + 1e-8
        class_means = [Z[Y==c, k].mean() if np.any(Y==c) else 0.0 for c in range(num_classes)]
        between = np.var(class_means)
        vals.append(float(between / overall))
    return torch.tensor(vals, dtype=torch.float32)

def main(args):
    train_loader, val_loader, test_loader, h_dim, num_classes, vocab = make_loaders(
        N_TRAIN=args.n_train, N_VAL=args.n_val, N_TEST=args.n_test, K=args.topk, batch=args.batch, seed=42)

    # ---------- Baseline ----------
    baseline = BaselineHead(h_dim=h_dim, num_classes=num_classes).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=args.lr)
    best_va, best_logits = -1, None
    for e in range(1, args.epochs+1):
        tr_loss, tr_acc, _, _ = run_epoch_baseline(baseline, train_loader, opt_b)
        va_loss, va_acc, va_L, va_Y = run_epoch_baseline(baseline, val_loader, None)
        print(f"[Baseline] epoch {e:02d} loss {tr_loss:.3f}/{va_loss:.3f} acc {tr_acc:.3f}/{va_acc:.3f}")
        if va_acc > best_va: best_va, best_logits = va_acc, (va_L, va_Y)
    b_L, b_Y = best_logits
    b_scaler = fit_temperature(b_L, b_Y)
    b_ece_pre, b_ece_post = ece_score(b_L, b_Y), ece_score(b_scaler(b_L), b_Y)
    print(f"[Baseline] ECE val pre→post: {b_ece_pre:.3f} → {b_ece_post:.3f}")
    # test baseline
    _, _, Lb_test, Yb_test = run_epoch_baseline(baseline, test_loader, None)
    Lb_test_cal = b_scaler(Lb_test)
    b_test_acc = (Lb_test_cal.argmax(1) == Yb_test).float().mean().item()
    b_test_ece = ece_score(Lb_test_cal, Yb_test)
    print(f"[Baseline] TEST acc={b_test_acc:.3f} ECE={b_test_ece:.3f}")

    # ---------- Lens ----------
    validity = estimate_cue_validity(train_loader, num_classes)
    print("Cue validity (unnorm):", validity.tolist())
    print("Cue validity (norm):  ", torch.softmax(validity, dim=0).tolist())

    lens = LensHead(h_dim=h_dim, num_classes=num_classes,
                    validity_prior=validity, lambda_validity=args.lmbd).to(device)
    opt_l = torch.optim.AdamW(lens.parameters(), lr=args.lr)
    best_va, best_logits = -1, None
    for e in range(1, args.epochs+1):
        tr_loss, tr_acc, _, _ = run_epoch_lens(lens, train_loader, opt_l)
        va_loss, va_acc, va_L, va_Y = run_epoch_lens(lens, val_loader, None)
        with torch.no_grad():
            u = torch.softmax(lens.u_params, dim=0).cpu().numpy()
        print(f"[Lens]     epoch {e:02d} loss {tr_loss:.3f}/{va_loss:.3f} acc {tr_acc:.3f}/{va_acc:.3f} u={np.round(u,3)}")
        if va_acc > best_va: best_va, best_logits = va_acc, (va_L, va_Y)
    l_L, l_Y = best_logits
    l_scaler = fit_temperature(l_L, l_Y)
    l_ece_pre, l_ece_post = ece_score(l_L, l_Y), ece_score(l_scaler(l_L), l_Y)
    print(f"[Lens]     ECE val pre→post: {l_ece_pre:.3f} → {l_ece_post:.3f}")
    # test lens
    _, _, Ll_test, Yl_test = run_epoch_lens(lens, test_loader, None)
    Ll_test_cal = l_scaler(Ll_test)
    l_test_acc = (Ll_test_cal.argmax(1) == Yl_test).float().mean().item()
    l_test_ece = ece_score(Ll_test_cal, Yl_test)
    print(f"[Lens]     TEST acc={l_test_acc:.3f} ECE={l_test_ece:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val",   type=int, default=400)
    ap.add_argument("--n_test",  type=int, default=400)
    ap.add_argument("--topk",    type=int, default=10)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--epochs",  type=int, default=8)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--lmbd",    type=float, default=0.1)  # validity regularizer
    args = ap.parse_args()
    main(args)
