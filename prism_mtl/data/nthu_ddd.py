"""NTHU-DDD two-class (normal / drowsy) static image dataset.

Expected layout::

    <data_root>/normal/**/*.jpg
    <data_root>/drowsy/**/*.jpg

Face crops are produced once with MediaPipe FaceMesh (100 px padding) and cached.
Images whose face is not detected fall back to the full frame.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

CLASS_TO_IDX = {"normal": 0, "drowsy": 1}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_root):
    rows = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = Path(data_root) / class_name
        if not class_dir.exists():
            raise FileNotFoundError(class_dir)
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                rows.append({"image_path": str(p), "class_name": class_name, "label": label})
    if not rows:
        raise RuntimeError(f"no images under {data_root}")
    return pd.DataFrame(rows)


def make_splits(data_root, seed):
    """Stratified image-level split: train 64 / val 16 / test 20."""
    df = collect_images(data_root)
    trainval, test = train_test_split(df, test_size=0.20, random_state=seed, shuffle=True, stratify=df["label"])
    train, val = train_test_split(trainval, test_size=0.20, random_state=seed, shuffle=True,
                                  stratify=trainval["label"])
    return {k: v.reset_index(drop=True) for k, v in (("train", train), ("val", val), ("test", test))}


def face_cache_path(cache_dir, image_path, class_name):
    p = Path(image_path)
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / class_name / f"{p.stem}_{h}.jpg"


def crop_face(img_bgr, face_mesh, pad=100):
    """Face box from the FaceMesh landmarks, padded; returns (crop, detected)."""
    H, W = img_bgr.shape[:2]
    result = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return img_bgr.copy(), False

    lm = result.multi_face_landmarks[0].landmark
    xs = np.array([p.x * W for p in lm], dtype=np.float32)
    ys = np.array([p.y * H for p in lm], dtype=np.float32)
    x1, y1 = max(0, int(np.floor(xs.min())) - pad), max(0, int(np.floor(ys.min())) - pad)
    x2, y2 = min(W, int(np.ceil(xs.max())) + pad), min(H, int(np.ceil(ys.max())) + pad)
    if x2 <= x1 or y2 <= y1:
        return img_bgr.copy(), False
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return img_bgr.copy(), False
    return crop, True


def build_face_cache(df, cache_dir, size=224, pad=100, min_conf=0.5):
    todo = [r for r in df.itertuples(index=False)
            if not face_cache_path(cache_dir, r.image_path, r.class_name).exists()]
    if not todo:
        return
    import mediapipe as mp

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=min_conf) as fm:
        for r in tqdm(todo, desc="FaceMesh cache"):
            out = face_cache_path(cache_dir, r.image_path, r.class_name)
            out.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(r.image_path)
            if img is None:
                raise RuntimeError(f"failed to read {r.image_path}")
            crop, _ = crop_face(img, fm, pad=pad)
            cv2.imwrite(str(out), cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))


def _read_rgb(path, size):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    img = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


class NTHUDDDDataset(Dataset):
    """Returns (face, original, label) with uint8 images of shape (3, 224, 224)."""

    def __init__(self, df, cache_dir, size=224):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        face_path = face_cache_path(self.cache_dir, row["image_path"], row["class_name"])
        if not face_path.exists():
            face_path = row["image_path"]
        return (_read_rgb(face_path, self.size), _read_rgb(row["image_path"], self.size), int(row["label"]))
