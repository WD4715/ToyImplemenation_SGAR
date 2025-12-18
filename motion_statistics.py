#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute mean/std for holistic and part motions separately.

Input:
  /mnt/e/research/motion_data/split_joints/
    ├── 00001/
    │    ├── holistic.npy
    │    ├── left_arm.npy
    │    ├── right_arm.npy
    │    ├── left_leg.npy
    │    ├── right_leg.npy
    │    └── torso.npy
    ├── 00002/
    │    └── ...

Output:
  /mnt/e/research/motion_data/
    ├── holistic_Mean.npy
    ├── holistic_Std.npy
    ├── left_arm_Mean.npy
    ├── left_arm_Std.npy
    ├── ...
"""

import os
import glob
import numpy as np
from tqdm import tqdm

# =========================
# Paths
# =========================

SPLIT_ROOT = "/mnt/e/research/motion_data/split_joints"
OUT_ROOT   = "/mnt/e/research/motion_data"

PARTS = [
    "holistic",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "torso",
]

os.makedirs(OUT_ROOT, exist_ok=True)


# =========================
# Utility
# =========================

def compute_mean_std_for_part(part_name: str):
    """
    Compute mean/std for a given part over all IDs.
    """
    print(f"\n{'='*80}")
    print(f"[INFO] Computing mean/std for part: {part_name}")
    print(f"{'='*80}")

    # Collect all npy paths for this part
    part_paths = []
    for id_dir in sorted(os.listdir(SPLIT_ROOT)):
        full_dir = os.path.join(SPLIT_ROOT, id_dir)
        if not os.path.isdir(full_dir):
            continue

        npy_path = os.path.join(full_dir, f"{part_name}.npy")
        if os.path.exists(npy_path):
            part_paths.append(npy_path)

    if len(part_paths) == 0:
        print(f"[WARN] No files found for part: {part_name}")
        return

    print(f"[INFO] Found {len(part_paths)} files")

    # First pass: compute mean
    sum_vec = None
    count = 0

    for p in tqdm(part_paths, desc=f"[Mean] {part_name}"):
        x = np.load(p)  # (T, J, 3)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Invalid shape {x.shape} in {p}")

        T, J, _ = x.shape
        x_flat = x.reshape(T, J * 3).astype(np.float64)  # 안정성

        if sum_vec is None:
            D = x_flat.shape[1]
            sum_vec = np.zeros(D, dtype=np.float64)

        sum_vec += x_flat.sum(axis=0)
        count += x_flat.shape[0]

    mean = sum_vec / max(count, 1)

    # Second pass: compute variance
    sq_sum = np.zeros_like(mean)

    for p in tqdm(part_paths, desc=f"[Std]  {part_name}"):
        x = np.load(p)
        T, J, _ = x.shape
        x_flat = x.reshape(T, J * 3).astype(np.float64)
        diff = x_flat - mean
        sq_sum += (diff ** 2).sum(axis=0)

    var = sq_sum / max(count, 1)
    std = np.sqrt(var)

    # 안정화
    std[std < 1e-8] = 1.0

    # Save
    mean_path = os.path.join(OUT_ROOT, f"{part_name}_Mean.npy")
    std_path  = os.path.join(OUT_ROOT, f"{part_name}_Std.npy")

    np.save(mean_path, mean.astype(np.float32))
    np.save(std_path,  std.astype(np.float32))

    print(f"[SAVE] Mean → {mean_path}  (D={mean.shape[0]})")
    print(f"[SAVE] Std  → {std_path}")


# =========================
# Main
# =========================

def main():
    for part in PARTS:
        compute_mean_std_for_part(part)

    print(f"\n{'='*80}")
    print("[DONE] All mean/std files generated successfully.")
    print(f"Saved under: {OUT_ROOT}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
