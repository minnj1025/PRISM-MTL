"""Train and evaluate the static PRISM-MTL adaptation on NTHU-DDD (normal vs. drowsy).

    python scripts/train_nthu_ddd.py --data_root /path/to/nthu_2class --face_cache /path/to/face_cache
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_mtl.data.nthu_ddd import NTHUDDDDataset, build_face_cache, collect_images, make_splits  # noqa: E402
from prism_mtl.models import PRISM_NTHU  # noqa: E402
from prism_mtl.utils import (cleanup_cuda, get_device, load_last, load_weights, make_loaders,  # noqa: E402
                             mean_std, save_last, seed_everything, write_json)

DEVICE = get_device()
CE = nn.CrossEntropyLoss()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_device(batch):
    face, original, y = batch
    norm = lambda x: (x.float().div(255.0).to(DEVICE, non_blocking=True) - MEAN.to(DEVICE)) / STD.to(DEVICE)  # noqa: E731
    return norm(face), norm(original), y.to(DEVICE, non_blocking=True)


def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        face, original, y = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=DEVICE == "cuda"):
            loss = CE(model(face, original), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, desc="eval"):
    model.eval()
    total, y_true, y_pred = 0.0, [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        face, original, y = to_device(batch)
        with autocast("cuda", enabled=DEVICE == "cuda"):
            logits = model(face, original)
            total += float(CE(logits, y).item())
        y_true += y.cpu().tolist()
        y_pred += logits.argmax(1).cpu().tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {"loss": total / max(1, len(loader)),
            "acc": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "normal_acc": float(per_class[0]), "drowsy_acc": float(per_class[1]),
            "confusion_matrix": cm.tolist()}


def run_seed(args, seed):
    seed_everything(seed)
    run_dir = Path(args.out) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(args.data_root, seed)
    for name, df in splits.items():
        df.to_csv(run_dir / f"{name}.csv", index=False)
    ds = [NTHUDDDDataset(splits[k], args.face_cache) for k in ("train", "val", "test")]
    train_loader, val_loader, test_loader = make_loaders(*ds, args.batch_size, args.num_workers, seed)
    model = PRISM_NTHU(heads=args.heads).to(DEVICE)

    if args.eval_ckpt:
        ckpt = args.eval_ckpt.format(seed=seed)
        load_weights(model, ckpt)
        return {"seed": seed, "ckpt": ckpt, **evaluate(model, test_loader, "test")}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=DEVICE == "cuda")
    best = {"val_loss": 1e9, "val_bal_acc": -1.0, "epoch_loss": None, "epoch_bal_acc": None}
    start = 1
    if (run_dir / "last.pt").exists():
        start, best = load_last(run_dir / "last.pt", model, optimizer, scaler)

    for epoch in range(start, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        va = evaluate(model, val_loader, f"val {epoch}")
        print(f"[seed {seed} | epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va['loss']:.4f} "
              f"val_acc={va['acc']:.4f} val_bal_acc={va['balanced_acc']:.4f}")
        state = {"epoch": epoch, "model": model.state_dict(), "val_loss": va["loss"], "val_macc": va["balanced_acc"]}
        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch_loss=epoch)
            torch.save(state, run_dir / "bestloss.pt")
        if va["balanced_acc"] > best["val_bal_acc"]:
            best.update(val_bal_acc=va["balanced_acc"], epoch_bal_acc=epoch)
            torch.save(state, run_dir / "bestmacc.pt")
        save_last(run_dir / "last.pt", epoch, model, optimizer, scaler, best)

    ck = load_weights(model, run_dir / "bestmacc.pt")
    result = {"seed": seed, "ckpt_epoch": ck.get("epoch"), **evaluate(model, test_loader, "test"), "best": best}
    write_json(result, run_dir / "done.json")
    del model, optimizer, scaler
    cleanup_cuda()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="folder with normal/ and drowsy/ images")
    p.add_argument("--face_cache", required=True, help="where MediaPipe face crops are cached")
    p.add_argument("--out", default="runs/nthu_ddd")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--eval_ckpt", default=None, help="only evaluate this checkpoint; {seed} is filled in")
    args = p.parse_args()

    build_face_cache(collect_images(args.data_root), args.face_cache)
    t0 = time.time()
    results = [run_seed(args, s) for s in args.seeds]
    for r in results:
        print(f"seed {r['seed']}: acc={r['acc'] * 100:.2f} macro_f1={r['macro_f1']:.4f}")
    if len(results) > 1:
        summary = {k: mean_std([r[k] for r in results]) for k in ("acc", "macro_f1", "balanced_acc")}
        for k, (m, s) in summary.items():
            print(f"{k:13s} {m * 100:.2f} ± {s * 100:.2f}")
        if not args.eval_ckpt:
            write_json({"args": vars(args), "per_seed": results, "mean_std": summary,
                        "elapsed_sec": time.time() - t0}, Path(args.out) / "summary.json")


if __name__ == "__main__":
    main()
