"""Train and evaluate the two-task PRISM-MTL adaptation on HDD (VBR + TCR).

    python scripts/train_hdd.py --data_root /path/to/hdd_data
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_mtl.data import hdd  # noqa: E402
from prism_mtl.models import PRISM_HDD  # noqa: E402
from prism_mtl.utils import (cleanup_cuda, get_device, load_last, load_weights, make_loaders,  # noqa: E402
                             mean_std, save_last, seed_everything, write_json)

DEVICE = get_device()
TCR_THRESHOLD = 0.5


def to_device(batch):
    rgb, pedal, steering, motion, signal, yvbr, ytcr = batch
    x = [(rgb / 255.0).to(DEVICE, non_blocking=True)] + [s.to(DEVICE, non_blocking=True)
                                                        for s in (pedal, steering, motion, signal)]
    return x, yvbr.to(DEVICE, non_blocking=True), ytcr.to(DEVICE, non_blocking=True)


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _ap(y_bin, score):
    return float(average_precision_score(y_bin, score)) if y_bin.sum() > 0 else None


def compute_metrics(y_vbr, p_vbr, y_tcr, s_tcr):
    y_vbr, p_vbr = np.asarray(y_vbr), np.asarray(p_vbr, dtype=np.float64)
    y_tcr, s_tcr = np.asarray(y_tcr, dtype=np.int32), np.asarray(s_tcr, dtype=np.float64)
    pred_vbr = p_vbr.argmax(1)
    pred_tcr = (s_tcr >= TCR_THRESHOLD).astype(np.int32)

    vbr_ap = [_ap((y_vbr == c).astype(np.int32), p_vbr[:, c]) for c in range(len(hdd.VBR_LABELS))]
    tcr_ap = [_ap(y_tcr[:, c], s_tcr[:, c]) for c in range(len(hdd.TCR_LABELS))]
    # A window with no context counts as a hit only if nothing is predicted.
    any_hit = [float(p.sum() == 0) if g.sum() == 0 else float((g * p).sum() > 0) for g, p in zip(y_tcr, pred_tcr)]

    m = {
        "vbr_acc": float(accuracy_score(y_vbr, pred_vbr)),
        "vbr_macro_f1": float(f1_score(y_vbr, pred_vbr, average="macro", zero_division=0)),
        "vbr_map_all": _mean(vbr_ap),
        "vbr_map_fg": _mean(vbr_ap[1:]),
        "tcr_map": _mean(tcr_ap),
        "tcr_micro_f1": float(f1_score(y_tcr, pred_tcr, average="micro", zero_division=0)),
        "tcr_macro_f1": float(f1_score(y_tcr, pred_tcr, average="macro", zero_division=0)),
        "tcr_any_hit_acc": float(np.mean(any_hit)),
    }
    m["macc"] = float(np.mean([m["vbr_acc"], m["tcr_any_hit_acc"]]))
    m["mean_map"] = _mean([m["vbr_map_fg"], m["tcr_map"]])
    return m


def run_epoch(model, loader, losses, optimizer=None, scaler=None, desc=""):
    """One pass; trains when an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total, y_vbr, p_vbr, y_tcr, s_tcr = 0.0, [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        x, yv, yt = to_device(batch)
        with torch.set_grad_enabled(train):
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=DEVICE == "cuda"):
                out_vbr, out_tcr = model(*x)
                loss = losses[0](out_vbr, yv) + losses[1](out_tcr, yt)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total += float(loss.item())
        if not train:
            y_vbr += yv.cpu().tolist()
            p_vbr += torch.softmax(out_vbr, 1).float().cpu().numpy().tolist()
            y_tcr += yt.float().cpu().numpy().tolist()
            s_tcr += torch.sigmoid(out_tcr).float().cpu().numpy().tolist()
    avg = total / max(1, len(loader))
    return avg if train else {"loss": avg, **compute_metrics(y_vbr, p_vbr, y_tcr, s_tcr)}


def run_seed(args, seed, paths, tcr_cache):
    seed_everything(seed)
    run_dir = Path(args.out) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    split_dir = hdd.make_splits(paths, seed, tcr_cache, Path(args.out) / f"splits_seed{seed}")
    mean, std = hdd.sensor_stats(paths, split_dir / "train.csv", split_dir / "sensor_stats_train.json")
    ds = [hdd.HDDWindowDataset(paths, split_dir / f"{k}.csv", mean, std) for k in ("train", "val", "test")]
    train_loader, val_loader, test_loader = make_loaders(*ds, args.batch_size, args.num_workers, seed)

    losses = (nn.CrossEntropyLoss(weight=hdd.vbr_class_weights(split_dir / "train.csv").to(DEVICE)),
              nn.BCEWithLogitsLoss(pos_weight=hdd.tcr_pos_weight(split_dir / "train.csv").to(DEVICE)))
    model = PRISM_HDD(num_vbr=len(hdd.VBR_LABELS), num_tcr=len(hdd.TCR_LABELS), heads=args.heads).to(DEVICE)

    if args.eval_ckpt:
        ckpt = args.eval_ckpt.format(seed=seed)
        load_weights(model, ckpt)
        return {"seed": seed, "ckpt": ckpt, **run_epoch(model, test_loader, losses, desc="test")}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=DEVICE == "cuda")
    best = {"val_loss": 1e9, "val_score": -1.0, "epoch_loss": None, "epoch_score": None}
    start = 1
    if (run_dir / "last.pt").exists():
        start, best = load_last(run_dir / "last.pt", model, optimizer, scaler)

    for epoch in range(start, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, losses, optimizer, scaler, f"train {epoch}")
        va = run_epoch(model, val_loader, losses, desc=f"val {epoch}")
        score = va["mean_map"] if va["mean_map"] is not None else -1.0
        print(f"[seed {seed} | epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va['loss']:.4f} "
              f"val_vbr_acc={va['vbr_acc']:.4f} val_mean_mAP={score:.4f}")
        state = {"epoch": epoch, "model": model.state_dict(), "val_loss": va["loss"], "val_score": score}
        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch_loss=epoch)
            torch.save(state, run_dir / "bestloss.pt")
        if score > best["val_score"]:
            best.update(val_score=score, epoch_score=epoch)
            torch.save(state, run_dir / "bestmacc.pt")
        save_last(run_dir / "last.pt", epoch, model, optimizer, scaler, best)

    ck = load_weights(model, run_dir / "bestmacc.pt")
    result = {"seed": seed, "ckpt_epoch": ck.get("epoch"),
              **run_epoch(model, test_loader, losses, desc="test"), "best": best}
    write_json(result, run_dir / "done.json")
    del model, optimizer, scaler
    cleanup_cuda()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", default="runs/hdd")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--eval_ckpt", default=None, help="only evaluate this checkpoint; {seed} is filled in")
    args = p.parse_args()

    paths = hdd.HDDPaths(args.data_root)
    tcr_cache = Path(args.out) / "tcr_cache"
    hdd.build_tcr_cache(paths, tcr_cache)

    t0 = time.time()
    results = [run_seed(args, s, paths, tcr_cache) for s in args.seeds]
    keys = ("vbr_acc", "vbr_map_all", "vbr_map_fg", "vbr_macro_f1", "tcr_any_hit_acc", "tcr_macro_f1", "tcr_map", "macc")
    for r in results:
        print(f"seed {r['seed']}: " + " ".join(f"{k}={r[k] * 100:.2f}" for k in keys))
    if len(results) > 1:
        summary = {k: mean_std([r[k] for r in results]) for k in keys}
        for k, (m, s) in summary.items():
            print(f"{k:16s} {m * 100:.2f} ± {s * 100:.2f}")
        if not args.eval_ckpt:
            write_json({"args": vars(args), "per_seed": results, "mean_std": summary,
                        "elapsed_sec": time.time() - t0}, Path(args.out) / "summary.json")


if __name__ == "__main__":
    main()
