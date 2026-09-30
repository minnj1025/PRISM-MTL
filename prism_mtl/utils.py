import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return "cuda"
    return "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The paper runs favour speed; results can differ slightly between GPUs.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0, worker_init_fn=seed_worker)
    return (DataLoader(train_ds, shuffle=True, generator=g, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common))


def _rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    if not st:
        return
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def save_last(path, epoch, model, optimizer, scaler, best):
    torch.save({"epoch": int(epoch), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best": best, "rng": _rng_state()}, path)


def load_last(path, model, optimizer, scaler):
    """Resume from last.pt; returns (start_epoch, best)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    if ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    _set_rng_state(ck.get("rng"))
    return int(ck["epoch"]) + 1, ck.get("best", {})


def load_weights(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    return ck


def mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
