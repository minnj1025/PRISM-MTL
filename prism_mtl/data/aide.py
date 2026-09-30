"""AIDE clip dataset.

Expected layout (the official AIDE release)::

    AIDE_Dataset/
      0001/{incarframes,frontframes,leftframes,rightframes}/*.jpg
      ...
      annotation/0001.json
"""
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

EMOTION_LABEL = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]
DRIVER_BEHAVIOR_LABEL = ["Smoking", "Making Phone", "Looking Around", "Dozing Off",
                         "Normal Driving", "Talking", "Body Movement"]
SCENE_CENTRIC_CONTEXT_LABEL = ["Traffic Jam", "Waiting", "Smooth Traffic"]
VEHICLE_BASED_CONTEXT_LABEL = ["Parking", "Turning", "Backward Moving", "Changing Lane", "Forward Moving"]

VIEWS = ("incarframes", "frontframes", "leftframes", "rightframes")


# ---------------------------------------------------------------- splits
def list_clips(data_root):
    """Clip ids that have both a frame folder and an annotation, in sorted order."""
    data_root = Path(data_root)
    ann_dir = data_root / "annotation"
    return [p.name for p in sorted(data_root.iterdir())
            if p.is_dir() and p.name.isdigit() and (ann_dir / f"{p.name}.json").exists()]


def make_splits(data_root, seed, train_ratio=0.8, val_ratio_in_train=0.2):
    """Clip-level 64/16/20 split used in the paper (train/val/test clip ids)."""
    df = pd.DataFrame({"clip_id": list_clips(data_root)})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_ratio)
    train_all, test = df.iloc[:n_train], df.iloc[n_train:]

    train_all = train_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(train_all) * val_ratio_in_train)
    val, train = train_all.iloc[:n_val], train_all.iloc[n_val:]
    return {"train": train.clip_id.tolist(), "val": val.clip_id.tolist(), "test": test.clip_id.tolist()}


def save_splits(splits, split_dir):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (split_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")


def load_splits(split_dir):
    split_dir = Path(split_dir)
    return {name: (split_dir / f"{name}.txt").read_text().split() for name in ("train", "val", "test")}


# ---------------------------------------------------------------- frames
def _base_candidate_indices(total_frames, stride, target):
    idxs = list(range(0, total_frames, stride))
    if len(idxs) < target and (total_frames - 1) not in idxs:
        idxs.append(total_frames - 1)
    return idxs


def _has_pose(item):
    r = item.get("result", None)
    return isinstance(r, list) and len(r) > 0


def select_valid_indices(pose_list, stride=3, target=16):
    """Every `stride`-th frame that has a detected driver, topped up to `target` frames."""
    valid = []
    for i in _base_candidate_indices(len(pose_list), stride, target):
        if i < len(pose_list) and _has_pose(pose_list[i]):
            valid.append(i)
        if len(valid) >= target:
            return valid[:target]

    for i in range(len(pose_list)):
        if len(valid) >= target:
            break
        if i not in valid and _has_pose(pose_list[i]):
            valid.append(i)

    if not valid:
        raise RuntimeError("no frame with a detected driver in this clip")
    while len(valid) < target:
        valid.append(valid[-1])
    return valid[:target]


def _to_rgb_tensor(img_bgr, size):
    img = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img)


class AIDEDataset(Dataset):
    """Returns uint8 tensors; divide images by 255 before the model.

    keypoints: (3, T, 136, 1)   context: (3, 4T, 224, 224)
    body, face: (3, T, 224, 224)
    labels: emotion, behavior, scene, vehicle
    """

    def __init__(self, data_root, clip_ids, T=16, stride=3, size=224,
                 face_pad=(200, 20, 100, 50)):
        self.data_root = Path(data_root)
        self.clip_ids = list(clip_ids)
        self.T, self.stride, self.size = T, stride, size
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = face_pad

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        cid = self.clip_ids[idx]
        clip_dir = self.data_root / cid
        ann = json.load(open(self.data_root / "annotation" / f"{cid}.json", "r", encoding="utf-8"))

        pose_list = ann["pose_list"]
        sel = select_valid_indices(pose_list, stride=self.stride, target=self.T)
        frames = {v: sorted((clip_dir / v).glob("*.jpg"), key=lambda p: int(p.stem)) for v in VIEWS}
        for v, files in frames.items():
            if len(files) < len(pose_list):
                raise RuntimeError(f"{clip_dir}/{v}: {len(files)} frames < {len(pose_list)} poses")

        context, body, face, kps = [], [], [], []
        for i in sel:
            img = cv2.imread(str(frames["incarframes"][i]))
            if img is None:
                raise RuntimeError(f"failed to read {frames['incarframes'][i]}")
            H, W, _ = img.shape
            res = pose_list[i]["result"][0]

            bx, by, bw, bh = map(int, res["bbox"])
            fx, fy, fw, fh = map(int, res["face_bbox"])
            img_body = img[max(0, by):min(H, by + bh), max(0, bx):min(W, bx + bw)]
            img_face = img[max(0, fy - self.pad_top):min(H, fy + fh + self.pad_bottom),
                           max(0, fx - self.pad_left):min(W, fx + fw + self.pad_right)]
            if img_body.size == 0:
                img_body = img
            if img_face.size == 0:
                img_face = img_body

            body.append(_to_rgb_tensor(img_body, self.size))
            face.append(_to_rgb_tensor(img_face, self.size))
            for v in VIEWS:
                im = cv2.imread(str(frames[v][i]))
                if im is None:
                    raise RuntimeError(f"failed to read {frames[v][i]}")
                context.append(_to_rgb_tensor(im, self.size))
            kps.append(torch.tensor(res["keypoints"], dtype=torch.float32).reshape(-1, 3))

        context = torch.stack(context).permute(3, 0, 1, 2).contiguous()
        body = torch.stack(body).permute(3, 0, 1, 2).contiguous()
        face = torch.stack(face).permute(3, 0, 1, 2).contiguous()
        keypoints = torch.stack(kps).permute(2, 0, 1).contiguous().unsqueeze(-1)

        vehicle = ann["vehicle_based_context_label"]
        if vehicle == "Forward":
            vehicle = "Forward Moving"
        return (keypoints, context, body, face,
                EMOTION_LABEL.index(ann["emotion_label"].capitalize()),
                DRIVER_BEHAVIOR_LABEL.index(ann["driver_behavior_label"]),
                SCENE_CENTRIC_CONTEXT_LABEL.index(ann["scene_centric_context_label"]),
                VEHICLE_BASED_CONTEXT_LABEL.index(vehicle))
