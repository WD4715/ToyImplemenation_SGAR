#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataloader.py
Dataset and DataLoader for split joints data with GAR text annotations
"""

import os
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import (
    SPLIT_JOINTS_DIR, DATA_ROOT, PART_NAMES, HOLISTIC_NAME,
    canonical_id, safe_text, load_gar_json_as_dict,
    _load_split_joint_npy, _pad_or_crop_TD
)


# ============================================================
# ID List Building
# ============================================================

def _has_required_npy(base_dir: str) -> bool:
    """Check if directory has all required npy files"""
    if not os.path.isdir(base_dir):
        return False
    if not os.path.exists(os.path.join(base_dir, f"{HOLISTIC_NAME}.npy")):
        return False
    for p in PART_NAMES:
        if not os.path.exists(os.path.join(base_dir, f"{p}.npy")):
            return False
    return True


def scan_all_ids(split_joints_dir: str, id_width: int) -> List[str]:
    """
    Scan split_joints directory and return all valid sample IDs.
    """
    ids = []
    if not os.path.isdir(split_joints_dir):
        raise FileNotFoundError(
            f"[ERROR] split_joints dir not found: {split_joints_dir}"
        )

    for name in os.listdir(split_joints_dir):
        full = os.path.join(split_joints_dir, name)
        if not os.path.isdir(full):
            continue
        
        cid = canonical_id(name, width=id_width)
        ok = _has_required_npy(full)
        
        if ok:
            ids.append(cid)
        else:
            # Fallback: try canonical path
            alt = os.path.join(split_joints_dir, cid)
            if alt != full and _has_required_npy(alt):
                ids.append(cid)

    ids = sorted(set(ids))
    return ids


def read_ids_from_txt(split_file: str, id_width: int) -> List[str]:
    """Read IDs from split txt file and canonicalize"""
    ids = []
    if not os.path.exists(split_file):
        return ids
    
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            cid = canonical_id(s, width=id_width)
            if cid:
                ids.append(cid)
    return ids


def build_id_list(
    split: str,
    split_joints_dir: str,
    data_root: str,
    id_width: int,
    prefer_txt: bool = True,
    debug: bool = True
) -> List[str]:
    """
    Build ID list for a split.
    Strategy:
      1) If prefer_txt and {split}.txt exists: use it (filtered by actual files)
      2) If txt yields too few samples: fallback to folder scan
      3) If no txt: use folder scan
    """
    split_file = os.path.join(data_root, f"{split}.txt")
    ids_txt = read_ids_from_txt(split_file, id_width=id_width) if prefer_txt else []

    # Filter txt IDs: only keep those with valid files
    ids_valid_txt = []
    if ids_txt:
        for cid in ids_txt:
            base1 = os.path.join(split_joints_dir, cid)
            if _has_required_npy(base1):
                ids_valid_txt.append(cid)
        ids_valid_txt = sorted(set(ids_valid_txt))

    ids_scan = scan_all_ids(split_joints_dir, id_width=id_width)

    # Heuristic: if txt exists but yields too few samples, merge with scan
    if ids_valid_txt:
        if len(ids_valid_txt) < max(50, int(0.1 * max(1, len(ids_scan)))):
            merged = sorted(set(ids_valid_txt) | set(ids_scan))
            if debug:
                print(
                    f"[WARN] {split}.txt yields too few valid ids: "
                    f"{len(ids_valid_txt)} vs scanned {len(ids_scan)}."
                )
                print(f"[WARN] Using merged ids (txt ∪ scan): {len(merged)}")
            return merged
        else:
            if debug:
                print(f"[INFO] Using ids from {split}.txt (valid): {len(ids_valid_txt)}")
            return ids_valid_txt

    # No valid txt: use scan
    if debug:
        if os.path.exists(split_file):
            print(
                f"[WARN] {split}.txt exists but no valid ids found. "
                f"Fallback to scanning folders."
            )
        else:
            print(f"[INFO] {split}.txt not found. Using scanning folders.")
        print(f"[INFO] Scanned valid ids: {len(ids_scan)}")
    
    return ids_scan


# ============================================================
# Dataset
# ============================================================

class HumanAct12SplitJointsGARDataset(Dataset):
    """
    Dataset for split joints with GAR text annotations.
    
    Returns:
      - holistic: (max_frames, D_h)
      - parts: dict {part: (max_frames, D_part)}
      - global_text: caption string
      - part_texts: dict {part: string}
    
    Normalization:
      - holistic uses holistic_Mean/Std
      - each part uses {part}_Mean/Std
    """
    
    def __init__(
        self,
        split: str,
        stats: Dict[str, Dict[str, np.ndarray]],
        max_frames: int,
        gar_json_path: str,
        id_width: int,
        prefer_txt_split: bool = True,
        require_text: bool = True,
        debug: bool = True,
    ):
        super().__init__()
        self.split = split
        self.id_width = id_width
        self.debug = debug

        # Build ID list
        self.ids = build_id_list(
            split=split,
            split_joints_dir=SPLIT_JOINTS_DIR,
            data_root=DATA_ROOT,
            id_width=id_width,
            prefer_txt=prefer_txt_split,
            debug=debug
        )

        self.stats = stats
        self.max_frames = max_frames
        self.part_names = list(PART_NAMES)

        # Load GAR dict
        self.gar_dict = load_gar_json_as_dict(gar_json_path, id_width=id_width)

        # Filter by text availability
        if require_text:
            before = len(self.ids)
            self.ids = [sid for sid in self.ids if sid in self.gar_dict]
            after = len(self.ids)
            if debug:
                print(f"[INFO] {split}: filtered by GAR json ({before} -> {after})")
                if after == 0:
                    print("[ERROR] After GAR filtering, dataset is empty.")
                    print("        => ID format mismatch. Check:")
                    print(f"           - folder ids example: {self._peek_folder_ids(10)}")
                    print(f"           - gar ids example: {list(self.gar_dict.keys())[:10]}")

        if debug:
            print(
                f"[INFO] Dataset(split={split}) "
                f"num_ids={len(self.ids)}, max_frames={self.max_frames}"
            )
            if len(self.ids) > 0:
                print(f"[INFO] Example ids[{split}] (first 10): {self.ids[:10]}")

    def _peek_folder_ids(self, k: int = 10) -> List[str]:
        """Get sample folder IDs for debugging"""
        try:
            names = [
                d for d in os.listdir(SPLIT_JOINTS_DIR)
                if os.path.isdir(os.path.join(SPLIT_JOINTS_DIR, d))
            ]
            names = sorted(names)[:k]
            return [canonical_id(n, width=self.id_width) for n in names]
        except Exception:
            return []

    def __len__(self):
        return len(self.ids)

    def _normalize(self, x_td: np.ndarray, name: str) -> np.ndarray:
        """Normalize using per-stream mean/std"""
        mean = self.stats[name]["mean"]
        std = self.stats[name]["std"]
        if x_td.shape[1] != mean.shape[0]:
            raise ValueError(
                f"[ERROR] Stat dim mismatch for {name}: "
                f"x has D={x_td.shape[1]}, mean/std D={mean.shape[0]}"
            )
        return (x_td - mean) / std

    def _resolve_base_dir(self, sid: str) -> str:
        """Resolve base directory for sample ID"""
        base = os.path.join(SPLIT_JOINTS_DIR, sid)
        if os.path.isdir(base):
            return base
        
        # Fallback: scan for matching folder
        try:
            for name in os.listdir(SPLIT_JOINTS_DIR):
                full = os.path.join(SPLIT_JOINTS_DIR, name)
                if not os.path.isdir(full):
                    continue
                if canonical_id(name, width=self.id_width) == sid:
                    return full
        except Exception:
            pass
        
        return base

    def _load_one(self, sid: str) -> Dict[str, Any]:
        """Load one sample"""
        base = self._resolve_base_dir(sid)

        # Load holistic
        hol = _load_split_joint_npy(os.path.join(base, f"{HOLISTIC_NAME}.npy"))
        hol = _pad_or_crop_TD(hol, self.max_frames)
        hol = self._normalize(hol, HOLISTIC_NAME)

        # Load parts
        parts = {}
        for p in self.part_names:
            arr = _load_split_joint_npy(os.path.join(base, f"{p}.npy"))
            arr = _pad_or_crop_TD(arr, self.max_frames)
            arr = self._normalize(arr, p)
            parts[p] = arr

        # Load text
        item = self.gar_dict.get(sid, {})
        caption = safe_text(item.get("caption", ""))

        part_desc = item.get("part_descriptions", {}) or {}
        if not isinstance(part_desc, dict):
            part_desc = {}

        part_texts = {}
        for p in self.part_names:
            part_texts[p] = safe_text(part_desc.get(p, ""))

        return {
            "id": sid,
            "holistic": torch.from_numpy(hol).float(),  # (T,Dh)
            "parts": {
                p: torch.from_numpy(parts[p]).float() 
                for p in self.part_names
            },
            "global_text": caption,
            "part_texts": part_texts,
        }

    def __getitem__(self, idx):
        return self._load_one(self.ids[idx])


# ============================================================
# Collate Function
# ============================================================

def collate_split_joints(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader"""
    ids = [b["id"] for b in batch]
    hol = torch.stack([b["holistic"] for b in batch], dim=0)

    parts = {}
    for p in PART_NAMES:
        parts[p] = torch.stack([b["parts"][p] for b in batch], dim=0)

    global_texts = [b["global_text"] for b in batch]

    part_texts = {}
    for p in PART_NAMES:
        part_texts[p] = [b["part_texts"][p] for b in batch]

    return {
        "ids": ids,
        "holistic": hol,  # (B,T,Dh)
        "parts": parts,  # dict part -> (B,T,Dp)
        "global_text": global_texts,
        "part_texts": part_texts,
    }