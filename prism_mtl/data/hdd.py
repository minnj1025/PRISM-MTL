"""Honda Research Institute Driving Dataset (HDD).

Uses the official pre-extracted release (3 fps) plus the EAF annotations::

    <root>/camera/<session>/00001.jpg ...   (20200710_images.tar.gz)
    <root>/sensor/<session>.npy             (8 CAN signals, 20200710_sensors.tar.gz)
    <root>/target/<session>.npy             (goal-oriented action per frame, 20200710_labels.tar.gz)
    <root>/release_2019_07_08/EAF/*.eaf     (release_2019_07_08.tar.gz)

Tasks
  VBR: 12 goal-oriented actions, one label per 16-frame window.
  TCR: 6 traffic-context classes from the EAF Cause/Attention tiers, multi-label per window
       (an all-zero vector means no context).
Sessions are split 70/15/15; windows are 16 frames with stride 16.
"""
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

VBR_LABELS = ["Background", "intersection passing", "left turn", "right turn", "left lane change",
              "right lane change", "left lane branch", "right lane branch", "crosswalk passing",
              "railroad passing", "merge", "U-turn"]
TCR_LABELS = ["traffic_control", "congestion", "vehicle_interaction", "pedestrian_context",
              "two_wheeler_context", "parked_or_road_obstacle"]
SENSOR_FEATURES = ["accel_pedal_angle_percent", "steer_angle", "steer_speed", "speed",
                   "brake_pedal_pressure_kpa", "left_turn_signal", "right_turn_signal", "yaw_degree_per_second"]
SENSOR_GROUPS = {"pedal": [0, 4], "steering": [1, 2], "motion": [3, 7], "signal": [5, 6]}

FPS = 3.0
T = 16
WINDOW_STRIDE = 16
MIN_NON_BACKGROUND_FRAMES = 2
MIN_TCR_POS_FRAMES = 2


class HDDPaths:
    def __init__(self, root):
        self.root = Path(root)
        self.camera = self.root / "camera"
        self.sensor = self.root / "sensor"
        self.target = self.root / "target"
        self.eaf = self.root / "release_2019_07_08" / "EAF"
        for p in (self.camera, self.sensor, self.target, self.eaf):
            if not p.exists():
                raise FileNotFoundError(p)


# ---------------------------------------------------------------- TCR labels from EAF
def normalize_tcr_label(label):
    s = (label or "").strip().lower()
    if not s:
        return None
    if "congestion" in s:
        return 1
    if "sign" in s or "red light" in s or "yellow light" in s or "traffic light" in s:
        return 0
    if "pedestrian" in s:
        return 3
    if "bicyclist" in s or "motorcyclist" in s:
        return 4
    if "parked" in s or "road work" in s or " park" in s or s.startswith("park"):
        return 5
    if "vehicle" in s or "cut-in" in s or "merging" in s or "turning" in s:
        return 2
    return None


def parse_eaf_intervals(eaf_path):
    """(start_ms, end_ms, class) for every Cause/Attention annotation that maps to a TCR class."""
    root = ET.parse(eaf_path).getroot()
    slots = {e.attrib["TIME_SLOT_ID"]: int(e.attrib["TIME_VALUE"])
             for e in root.iter() if e.tag.split("}")[-1] == "TIME_SLOT"}

    intervals = []
    for tier in root.iter():
        if tier.tag.split("}")[-1] != "TIER":
            continue
        tier_id = tier.attrib.get("TIER_ID", "").lower()
        if "cause" not in tier_id and "attention" not in tier_id:
            continue
        for ann in tier.iter():
            if ann.tag.split("}")[-1] != "ALIGNABLE_ANNOTATION":
                continue
            start = slots.get(ann.attrib.get("TIME_SLOT_REF1"))
            end = slots.get(ann.attrib.get("TIME_SLOT_REF2"))
            if start is None or end is None or end <= start:
                continue
            value = ""
            for child in ann.iter():
                if child.tag.split("}")[-1] == "ANNOTATION_VALUE":
                    value = (child.text or "").strip()
                    break
            cls = normalize_tcr_label(value)
            if cls is not None:
                intervals.append((start, end, cls))
    return intervals


def _eaf_map(session_ids, eaf_dir):
    eaf_paths = sorted(Path(eaf_dir).glob("*.eaf"))

    def sid_of(p):
        parts = p.stem.split("-")
        return p.stem if len(parts) <= 1 else "".join(parts[:-1])

    direct = {sid_of(p): p for p in eaf_paths}
    out = {}
    for sid in session_ids:
        if sid in direct:
            out[sid] = direct[sid]
        else:
            matched = [p for p in eaf_paths if sid in p.stem or p.stem in sid]
            if matched:
                out[sid] = matched[0]
    return out


def build_tcr_cache(paths, cache_dir):
    """Per-session (num_frames, 6) multi-hot TCR arrays, built once from the EAF files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(paths.target.glob("*.npy"))
    if all((cache_dir / t.name).exists() for t in targets):
        return
    eaf_by_sid = _eaf_map([t.stem for t in targets], paths.eaf)
    for t in tqdm(targets, desc="TCR cache"):
        n = int(np.load(t, mmap_mode="r").shape[0])
        y = np.zeros((n, len(TCR_LABELS)), dtype=np.uint8)
        eaf = eaf_by_sid.get(t.stem)
        if eaf is not None:
            for start_ms, end_ms, cls in parse_eaf_intervals(eaf):
                s = max(0, int(np.floor(start_ms / 1000.0 * FPS)))
                e = min(n, int(np.ceil(end_ms / 1000.0 * FPS)))
                if e > s:
                    y[s:e, cls] = 1
        np.save(cache_dir / t.name, y)


# ---------------------------------------------------------------- windows / splits
def available_sessions(paths):
    return [t.stem for t in sorted(paths.target.glob("*.npy"))
            if (paths.sensor / t.name).exists() and (paths.camera / t.stem).is_dir()]


def session_splits(sessions, seed, train_ratio=0.70, val_ratio=0.15):
    sessions = list(sessions)
    random.Random(seed).shuffle(sessions)
    n_train, n_val = int(len(sessions) * train_ratio), int(len(sessions) * val_ratio)
    return (sorted(sessions[:n_train]), sorted(sessions[n_train:n_train + n_val]),
            sorted(sessions[n_train + n_val:]))


def _window_vbr(labels):
    labels = np.asarray(labels, dtype=np.int64)
    non_bg = labels[labels != 0]
    if len(non_bg) >= MIN_NON_BACKGROUND_FRAMES:
        return int(np.bincount(non_bg).argmax())
    return int(np.bincount(labels).argmax())


def make_windows(paths, sessions, tcr_cache_dir):
    rows = []
    for sid in sessions:
        vbr = np.load(paths.target / f"{sid}.npy", mmap_mode="r")
        tcr = np.load(Path(tcr_cache_dir) / f"{sid}.npy", mmap_mode="r")
        sensor = np.load(paths.sensor / f"{sid}.npy", mmap_mode="r")
        n = min(vbr.shape[0], tcr.shape[0], sensor.shape[0], len(list((paths.camera / sid).glob("*.jpg"))))
        for start in range(0, n - T + 1, WINDOW_STRIDE):
            tcr_multi = (np.asarray(tcr[start:start + T], dtype=np.uint8).sum(axis=0) >= MIN_TCR_POS_FRAMES)
            rows.append({
                "session_id": sid, "start": start, "end": start + T,
                "vbr": _window_vbr(vbr[start:start + T]),
                "tcr_multi": json.dumps(tcr_multi.astype(int).tolist(), separators=(",", ":")),
            })
    return pd.DataFrame(rows)


def make_splits(paths, seed, tcr_cache_dir, split_dir):
    """Window CSVs for train/val/test; reused if they already exist."""
    split_dir = Path(split_dir)
    if all((split_dir / f"{n}.csv").exists() for n in ("train", "val", "test")):
        return split_dir
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, sessions in zip(("train", "val", "test"), session_splits(available_sessions(paths), seed)):
        make_windows(paths, sessions, tcr_cache_dir).to_csv(split_dir / f"{name}.csv", index=False)
    return split_dir


def sensor_stats(paths, train_csv, out_path):
    """Per-feature mean/std over every frame of the training sessions."""
    out_path = Path(out_path)
    if out_path.exists():
        s = json.load(open(out_path))
        return np.asarray(s["mean"], np.float32), np.asarray(s["std"], np.float32)
    total, sums, sumsq = 0, np.zeros(8), np.zeros(8)
    for sid in sorted(pd.read_csv(train_csv)["session_id"].astype(str).unique()):
        x = np.asarray(np.load(paths.sensor / f"{sid}.npy", mmap_mode="r"), dtype=np.float64)
        total += x.shape[0]
        sums += x.sum(axis=0)
        sumsq += (x * x).sum(axis=0)
    mean = sums / max(1, total)
    std = np.sqrt(np.maximum(sumsq / max(1, total) - mean * mean, 1e-8))
    std[std < 1e-6] = 1.0
    json.dump({"feature_order": SENSOR_FEATURES, "mean": mean.tolist(), "std": std.tolist()},
              open(out_path, "w"), indent=2)
    return mean.astype(np.float32), std.astype(np.float32)


def vbr_class_weights(train_csv):
    counts = np.bincount(pd.read_csv(train_csv)["vbr"].values.astype(np.int64),
                         minlength=len(VBR_LABELS)).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def tcr_pos_weight(train_csv):
    mat = np.asarray([json.loads(s) for s in pd.read_csv(train_csv)["tcr_multi"]], dtype=np.float32)
    pos = mat.sum(axis=0)
    w = np.clip(np.sqrt((mat.shape[0] - pos) / np.maximum(pos, 1.0)), 0.25, 10.0)
    return torch.tensor(w, dtype=torch.float32)


class HDDWindowDataset(Dataset):
    """Returns rgb (3, T, 224, 224) uint8, four standardized sensor groups (T, 2), vbr, tcr (6,)."""

    def __init__(self, paths, csv_file, sensor_mean, sensor_std, size=224):
        self.paths = paths
        self.df = pd.read_csv(csv_file)
        self.size = size
        self.mean = np.asarray(sensor_mean, np.float32)
        self.std = np.asarray(sensor_std, np.float32)
        self._sensor = {}

    def __len__(self):
        return len(self.df)

    def _read(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"failed to read {path}")
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid, start, end = str(row["session_id"]), int(row["start"]), int(row["end"])
        rgb = torch.stack([self._read(self.paths.camera / sid / f"{i + 1:05d}.jpg")
                           for i in range(start, end)]).permute(3, 0, 1, 2).contiguous()

        if sid not in self._sensor:
            self._sensor[sid] = np.asarray(np.load(self.paths.sensor / f"{sid}.npy", mmap_mode="r"), np.float32)
        s = (self._sensor[sid][start:end] - self.mean) / self.std
        groups = [torch.from_numpy(s[:, SENSOR_GROUPS[g]].astype(np.float32))
                  for g in ("pedal", "steering", "motion", "signal")]
        tcr = torch.tensor(json.loads(row["tcr_multi"]), dtype=torch.float32)
        return (rgb, *groups, int(row["vbr"]), tcr)
