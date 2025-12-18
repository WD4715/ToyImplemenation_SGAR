#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
Utility functions: ID canonicalization, stats loading, logging, etc.
"""

import os
import re
import json
import time
import random
from typing import Any, Dict, List

import numpy as np
import torch


# ============================================================
# Constants
# ============================================================

DATA_ROOT = "/mnt/e/research/motion_data"
SPLIT_JOINTS_DIR = os.path.join(DATA_ROOT, "split_joints")
TEXT_DIR = os.path.join(DATA_ROOT, "texts")

PART_NAMES = ["right_arm", "left_arm", "right_leg", "left_leg", "torso"]
HOLISTIC_NAME = "holistic"

MOT_PATH = (
    "/mnt/c/Users/AIV/code/GAR_pose/"
    "MotionDiffuse/checkpoints/humanml_trans_dec_512_bert/model000600000.pt"
)


# ============================================================
# Reproducibility / Logging
# ============================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    """Get current timestamp string"""
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def ensure_dir(p: str):
    """Create directory if it doesn't exist"""
    if p:
        os.makedirs(p, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    """Save dictionary as JSON file"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_text(s: Any) -> str:
    """Convert to safe text string, handling None and 'undefined'"""
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    s2 = s.strip()
    if s2.lower() == "undefined":
        return ""
    return s2


# ============================================================
# ID Canonicalization
# ============================================================

def canonical_id(x: Any, width: int = 5) -> str:
    """
    Extract digits only and zfill to width.
    Examples:
      "1" -> "00001"
      1 -> "00001"
      "00001" -> "00001"
      "id: 1369" -> "01369"
    """
    if x is None:
        return ""
    s = str(x).strip()
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return ""
    # keep last `width` digits if longer, else zfill
    if len(digits) > width:
        digits = digits[-width:]
    return digits.zfill(width)


def infer_id_width_from_dir(split_joints_dir: str) -> int:
    """
    Infer ID width from folder names.
    Returns the most common folder name length, or 5 as fallback.
    """
    try:
        names = [d for d in os.listdir(split_joints_dir) 
                 if os.path.isdir(os.path.join(split_joints_dir, d))]
        if not names:
            return 5
        # Get mode of lengths
        lens = [len(n) for n in names]
        return max(set(lens), key=lens.count)
    except Exception:
        return 5


# ============================================================
# Stats Loading
# ============================================================

def load_stats_dict(root: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load per-stream mean/std statistics.
    Returns:
      stats[name] = {"mean": (D,), "std": (D,)}
    Files:
      {name}_Mean.npy, {name}_Std.npy
    """
    names = [HOLISTIC_NAME] + PART_NAMES
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    
    for name in names:
        mean_path = os.path.join(root, f"{name}_Mean.npy")
        std_path = os.path.join(root, f"{name}_Std.npy")
        
        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            raise FileNotFoundError(
                f"[ERROR] Missing stats for {name}: {mean_path}, {std_path}"
            )

        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        std[std == 0] = 1.0

        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(
                f"[ERROR] stats must be 1D vectors: {name}, "
                f"mean={mean.shape}, std={std.shape}"
            )
        if mean.shape[0] != std.shape[0]:
            raise ValueError(
                f"[ERROR] mean/std dim mismatch: {name}, "
                f"mean={mean.shape}, std={std.shape}"
            )

        stats[name] = {"mean": mean, "std": std}

    print("[INFO] Loaded per-stream mean/std:")
    for k in stats:
        print(f"  - {k:10s}: D={stats[k]['mean'].shape[0]}")
    
    return stats


# ============================================================
# GAR JSON Loading
# ============================================================

def load_gar_json_as_dict(json_path: str, id_width: int) -> Dict[str, Dict[str, Any]]:
    """
    Load GAR JSON file and canonicalize IDs.
    Returns dict mapping canonical_id -> item
    """
    if not json_path or not os.path.exists(json_path):
        print(f"[WARN] GAR json not found: {json_path}")
        return {}
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    d: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, dict):
        # dict format: canonicalize keys
        for k, v in data.items():
            if isinstance(v, dict):
                cid = canonical_id(k, width=id_width)
                if cid:
                    d[cid] = v
        return d

    # list format: extract 'id' field
    for item in data:
        if not isinstance(item, dict):
            continue
        if "id" not in item:
            continue
        cid = canonical_id(item["id"], width=id_width)
        if not cid:
            continue
        d[cid] = item

    return d


# ============================================================
# Motion Data Utils
# ============================================================

def _pad_or_crop_TD(x: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad or crop motion tensor to max_frames"""
    T, D = x.shape
    if T >= max_frames:
        return x[:max_frames]
    pad = np.zeros((max_frames - T, D), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


def _load_split_joint_npy(npy_path: str) -> np.ndarray:
    """
    Load split joint npy file.
    Input: (T, J, 3)
    Output: (T, J*3) float32
    """
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(
            f"[ERROR] Expected (T,J,3) but got {arr.shape} at: {npy_path}"
        )
    T, J, C = arr.shape
    if C != 3:
        raise ValueError(
            f"[ERROR] Expected last dim=3 but got {arr.shape} at: {npy_path}"
        )
    arr = arr.reshape(T, J * 3).astype(np.float32)
    return arr


# ============================================================
# Loss Functions
# ============================================================

def contrastive_loss(z_t: torch.Tensor, z_m: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss"""
    import torch.nn.functional as F
    
    B = z_t.size(0)
    logits = (z_t @ z_m.t()) / tau
    labels = torch.arange(B, device=z_t.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def soft_contrastive_loss(
    z_m: torch.Tensor,
    z_t: torch.Tensor,
    tau: float = 0.07,
    tau_prime: float = 0.05
) -> torch.Tensor:
    """Soft contrastive loss (GAR style)"""
    import torch.nn.functional as F
    
    logits_tt = (z_t @ z_t.t()) / tau_prime
    p_soft = F.softmax(logits_tt, dim=1)

    logits_mt = (z_m @ z_t.t()) / tau
    log_p = F.log_softmax(logits_mt, dim=1)
    loss_t2m = -(p_soft * log_p).sum(dim=1).mean()

    logits_mm = (z_m @ z_m.t()) / tau_prime
    p_soft_m = F.softmax(logits_mm, dim=1)

    logits_tm = (z_t @ z_m.t()) / tau
    log_p_m = F.log_softmax(logits_tm, dim=1)
    loss_m2t = -(p_soft_m * log_p_m).sum(dim=1).mean()

    return 0.5 * (loss_t2m + loss_m2t)


# ============================================================
# Metrics
# ============================================================

def compute_retrieval_metrics(z_q: torch.Tensor, z_k: torch.Tensor):
    """
    Compute retrieval metrics (R@1, R@5, R@10, MR, MedR)
    
    Args:
        z_q: query embeddings (N, D)
        z_k: key embeddings (N, D)
    
    Returns:
        R1, R5, R10, MR, MedR
    """
    z_q = z_q.detach().cpu()
    z_k = z_k.detach().cpu()
    sims = (z_q @ z_k.t()).numpy()
    N = sims.shape[0]
    
    ranks = []
    for i in range(N):
        order = np.argsort(-sims[i])
        rank = np.where(order == i)[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    R1 = np.mean(ranks == 1)
    R5 = np.mean(ranks <= 5)
    R10 = np.mean(ranks <= 10)
    MR = np.mean(ranks)
    MedR = np.median(ranks)
    
    return R1, R5, R10, MR, MedR