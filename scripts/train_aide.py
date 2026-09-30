"""Train and evaluate PRISM-MTL on AIDE.

Main result (5 seeds):
    python scripts/train_aide.py --data_root /path/to/AIDE_Dataset

Ablations (seed 42):
    --modalities scene post gest     Table 8 (modality groups)
    --fusion agnostic                Table 12 (task-agnostic attention)
    --heads 1|2|4|8                  Table 13 (TSMF heads)

Evaluate a checkpoint only:
    python scripts/train_aide.py --data_root ... --seeds 42 --eval_ckpt runs/.../bestmacc.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_mtl.data.aide import AIDEDataset, load_splits, make_splits, save_splits  # noqa: E402
from prism_mtl.models import AIDE_TASKS, MODALITIES, PRISM_MTL  # noqa: E402
from prism_mtl.utils import (cleanup_cuda, get_device, load_last, load_weights, make_loaders,  # noqa: E402
                             mean_std, save_last, seed_everything, write_json)

REPO = Path(__file__).resolve().parents[1]
DEVICE = get_device()
CE = nn.CrossEntropyLoss()


def to_device(batch):
    kp, ctx, body, face, *ys = batch
    x = (kp.to(DEVICE, non_blocking=True),
         (ctx / 255.0).to(DEVICE, non_blocking=True),
         (body / 255.0).to(DEVICE, non_blocking=True),
         (face / 255.0).to(DEVICE, non_blocking=True))
    return x, [y.to(DEVICE, non_blocking=True) for y in ys]


def multitask_loss(out, ys):
    return sum(CE(o, y) for o, y in zip(out, ys))


def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total = 0.0
    for it, batch in enumerate(tqdm(loader, desc=f"train {epoch}", leave=False), start=1):
        x, ys = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=DEVICE == "cuda"):
            loss = multitask_loss(model(*x), ys)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, desc="eval"):
    """Returns total loss plus per-task accuracy / macro-F1 and their means."""
    model.eval()
    total, y_true, y_pred = 0.0, [[] for _ in AIDE_TASKS], [[] for _ in AIDE_TASKS]
    for batch in tqdm(loader, desc=desc, leave=False):
        x, ys = to_device(batch)
        with autocast("cuda", enabled=DEVICE == "cuda"):
            out = model(*x)
            total += float(multitask_loss(out, ys).item())
        for i in range(len(AIDE_TASKS)):
            y_true[i] += ys[i].cpu().tolist()
            y_pred[i] += out[i].argmax(1).cpu().tolist()

    m = {"loss": total / max(1, len(loader))}
    for i, t in enumerate(AIDE_TASKS):
        m[f"acc_{t}"] = accuracy_score(y_true[i], y_pred[i])
        m[f"f1_{t}"] = f1_score(y_true[i], y_pred[i], average="macro")
    m["macc"] = float(np.mean([m[f"acc_{t}"] for t in AIDE_TASKS]))
    m["mf1"] = float(np.mean([m[f"f1_{t}"] for t in AIDE_TASKS]))
    return m


def build_model(args):
    return PRISM_MTL(modalities=args.modalities, fusion=args.fusion, embed=args.embed,
                     d_model=args.d_model, heads=args.heads, dropout=0.1).to(DEVICE)


def get_splits(args, seed):
    split_dir = REPO / "splits" / "aide" / f"seed{seed}"
    if split_dir.exists() and not args.regenerate_splits:
        return load_splits(split_dir)
    splits = make_splits(args.data_root, seed)
    save_splits(splits, Path(args.out) / f"seed{seed}" / "splits")
    return splits


def run_seed(args, seed):
    seed_everything(seed)
    run_dir = Path(args.out) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = get_splits(args, seed)
    ds = [AIDEDataset(args.data_root, splits[k]) for k in ("train", "val", "test")]
    train_loader, val_loader, test_loader = make_loaders(*ds, args.batch_size, args.num_workers, seed)
    model = build_model(args)

    if args.eval_ckpt:
        ckpt = args.eval_ckpt.format(seed=seed)
        load_weights(model, ckpt)
        return {"seed": seed, "ckpt": ckpt, **evaluate(model, test_loader, "test")}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=DEVICE == "cuda")
    best = {"val_loss": 1e9, "val_macc": -1.0, "epoch_loss": None, "epoch_macc": None}
    start = 1
    if (run_dir / "last.pt").exists():
        start, best = load_last(run_dir / "last.pt", model, optimizer, scaler)

    for epoch in range(start, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        va = evaluate(model, val_loader, f"val {epoch}")
        print(f"[seed {seed} | epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va['loss']:.4f} "
              f"val_mAcc={va['macc']:.4f} val_mF1={va['mf1']:.4f}")
        state = {"epoch": epoch, "model": model.state_dict(), "val_loss": va["loss"], "val_macc": va["macc"]}
        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch_loss=epoch)
            torch.save(state, run_dir / "bestloss.pt")
        if va["macc"] > best["val_macc"]:
            best.update(val_macc=va["macc"], epoch_macc=epoch)
            torch.save(state, run_dir / "bestmacc.pt")
        save_last(run_dir / "last.pt", epoch, model, optimizer, scaler, best)

    # Test with the checkpoint that has the best validation mAcc.
    ck = load_weights(model, run_dir / "bestmacc.pt")
    result = {"seed": seed, "ckpt_epoch": ck.get("epoch"), **evaluate(model, test_loader, "test"), "best": best}
    write_json(result, run_dir / "done.json")
    del model, optimizer, scaler
    cleanup_cuda()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", default="runs/aide")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    p.add_argument("--fusion", default="tsmf", choices=["tsmf", "agnostic"])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embed", type=int, default=128)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--regenerate_splits", action="store_true",
                   help="rebuild splits from the data instead of using splits/aide/seedN")
    p.add_argument("--eval_ckpt", default=None, help="only evaluate this checkpoint on the test split; {seed} is filled in")
    args = p.parse_args()

    t0 = time.time()
    results = [run_seed(args, s) for s in args.seeds]
    for r in results:
        print(f"seed {r['seed']}: mAcc={r['macc'] * 100:.2f} " +
              " ".join(f"{t}={r[f'acc_{t}'] * 100:.2f}" for t in AIDE_TASKS))
    if len(results) > 1:
        summary = {k: mean_std([r[k] for r in results])
                   for k in ["macc", "mf1"] + [f"acc_{t}" for t in AIDE_TASKS]}
        for k, (m, s) in summary.items():
            print(f"{k:14s} {m * 100:.2f} ± {s * 100:.2f}")
        if not args.eval_ckpt:
            write_json({"args": vars(args), "per_seed": results, "mean_std": summary,
                        "elapsed_sec": time.time() - t0}, Path(args.out) / "summary.json")


if __name__ == "__main__":
    main()
