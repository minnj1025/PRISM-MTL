# PRISM-MTL

Official PyTorch implementation of

> **PRISM-MTL: Inter-Modal Selective Multi-Task Learning for Assistive Driving Perception**
> Minjun Kim, Gyuho Choi. *Mathematics* 2026, 14(15), 2812.
> [doi:10.3390/math14152812](https://doi.org/10.3390/math14152812)

PRISM-MTL recognizes the driver's emotion (DER) and behavior (DBR) together with the traffic context (TCR) and vehicle behavior (VBR) from multi-view video and body keypoints.

- **HSA-Net** encodes the face, body, and scene frames. It is a ResNet-18 with a different attention operator after each stage: identity, then directional spatial attention, then spatial attention, then channel attention.
- **Token-SE** encodes the posture and gesture keypoints with a frame-wise MLP, then re-weights the embedding channels over the whole clip.
- **TSMF** (Task-Specific Modality Fusion) gives each task its own learnable token. Each token cross-attends to the five modality tokens at every time step, so each task selects the modalities it needs.

## Installation

```bash
git clone https://github.com/minnj1025/PRISM-MTL.git
cd PRISM-MTL
pip install -r requirements.txt
```

The experiments used Python 3.10, PyTorch 2.9.1 (CUDA 12.8), and an RTX 4090.

## Data

No dataset is included in this repository. Download each dataset from its provider.

| Dataset | Expected layout |
|---|---|
| [AIDE](https://github.com/ydk122024/AIDE) | `AIDE_Dataset/<clip>/{incarframes,frontframes,leftframes,rightframes}/*.jpg`, `AIDE_Dataset/annotation/<clip>.json` |
| NTHU-DDD | `<root>/normal/*.jpg`, `<root>/drowsy/*.jpg` (frames extracted from the videos) |
| [HDD](https://usa.honda-ri.com/hdd) | `<root>/{camera,sensor,target}` from the pre-extracted release (`20200710_*`), and `<root>/release_2019_07_08/EAF` |

The AIDE clip splits used in the paper (seeds 42–46, train/val/test = 1855/463/580 clips) are in `splits/aide/`. They are identical to the splits produced by `prism_mtl.data.aide.make_splits`.

## Usage

### AIDE (main result, 5 seeds)

```bash
python scripts/train_aide.py --data_root /path/to/AIDE_Dataset
```

Settings: 50 epochs, batch size 8, AdamW (lr 1e-4, weight decay 1e-4), mixed precision, T = 16 frames with stride 3, and 4 TSMF heads. The checkpoint with the best validation mAcc is used for testing.

### Ablations on AIDE (seed 42)

| Paper | Command |
|---|---|
| Table 8, modality groups | `--seeds 42 --modalities scene post gest` (any subset of `face body scene post gest`) |
| Table 12, task-agnostic attention | `--seeds 42 --fusion agnostic` |
| Table 13, TSMF heads | `--seeds 42 --heads 1` (or 2, 4, 8) |

### NTHU-DDD and HDD

```bash
python scripts/train_nthu_ddd.py --data_root /path/to/nthu_2class --face_cache /path/to/face_cache
python scripts/train_hdd.py --data_root /path/to/hdd_data
```

- **NTHU-DDD**
  - Input is a MediaPipe face crop and the full frame.
  - The dataset is split at the image level into 64/16/20 with stratification.
  - Training uses batch size 32 with ImageNet normalization.
- **HDD**
  - Input is the front camera and four CAN sensor groups: pedal, steering, motion, and turn signal.
  - VBR uses the 12 goal-oriented actions. TCR uses 6 traffic-context classes derived from the EAF Cause/Attention tiers as a multi-label target.
  - The dataset is split at the session level into 70/15/15. Windows are 16 frames with stride 16.
  - Training uses batch size 16.

To evaluate a trained model without training, add `--eval_ckpt path/to/seed{seed}/bestmacc.pt` to any script.

## Results

Five seeds (42–46), mean ± standard deviation over the seeds.

**AIDE**

| DER Acc | DBR Acc | TCR Acc | VBR Acc | mAcc | mF1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 84.55 ± 0.91 | 80.59 ± 1.67 | 94.07 ± 0.80 | 85.79 ± 0.83 | **86.25 ± 0.35** | 80.17 ± 0.95 |

**NTHU-DDD**

| Accuracy | Macro-F1 |
|:---:|:---:|
| 96.47 ± 0.21 | 0.964 ± 0.002 |

To check these numbers, load the paper checkpoints into this code with `--eval_ckpt`. Each seed gives the same test score as the original run. The same holds for the checkpoints of the Table 8, 12, and 13 ablations.

Training uses `cudnn.benchmark` and mixed precision for speed, so results are not bit-exact across runs. Even the same checkpoint can occasionally flip one borderline test sample between evaluations.

## Repository structure

```
prism_mtl/
  hsa.py        HSA-Net (stage-wise attention ResNet-18)
  encoders.py   Token-SE keypoint and sensor encoders
  fusion.py     TSMF, its static single-task form, and the task-agnostic baseline
  models.py     PRISM_MTL (AIDE), PRISM_NTHU, PRISM_HDD
  data/         AIDE, NTHU-DDD and HDD datasets and splits
scripts/        training and evaluation entry points
splits/aide/    clip ids of the seed 42-46 splits
```

## Citation

```bibtex
@article{kim2026prismmtl,
  title   = {PRISM-MTL: Inter-Modal Selective Multi-Task Learning for Assistive Driving Perception},
  author  = {Kim, Minjun and Choi, Gyuho},
  journal = {Mathematics},
  volume  = {14},
  number  = {15},
  pages   = {2812},
  year    = {2026},
  doi     = {10.3390/math14152812}
}
```

## License

The code is released under the MIT License. Each dataset is governed by its own license.
