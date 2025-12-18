#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization.py
Attention visualization and qualitative analysis
"""

import os
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils import PART_NAMES, ensure_dir


# ============================================================
# Attention Visualization
# ============================================================

def save_attention_maps_probe(
    attn: torch.Tensor,
    T: int,
    num_parts: int,
    out_path_prefix: str
):
    """
    Save attention maps from probe.
    
    Args:
        attn: (B, heads, L, L) where L = 1 + num_parts + T
        T: number of time tokens
        num_parts: number of part tokens
        out_path_prefix: output path prefix
    """
    if attn is None:
        print("[WARN] attention is None. Skip attention visualization.")
        return

    ensure_dir(os.path.dirname(out_path_prefix))

    # Average over batch and heads
    A = attn.mean(dim=0).mean(dim=0)  # (L, L)
    L = A.shape[0]
    motion_start = 1 + num_parts
    motion_end = motion_start + T
    if motion_end > L:
        motion_end = L

    # CLS -> motion tokens
    cls_to_motion = A[0, motion_start:motion_end].detach().cpu().numpy()
    plt.figure(figsize=(10, 3))
    plt.imshow(cls_to_motion[None, :], aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.yticks([0], ["CLS"])
    plt.xlabel("Time token index")
    plt.title("Probe Attention: CLS -> MotionTokens")
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_probe_attn_cls.png", dpi=150)
    plt.close()

    # Part CLS -> motion tokens
    for j in range(num_parts):
        part_to_motion = A[1 + j, motion_start:motion_end].detach().cpu().numpy()
        plt.figure(figsize=(10, 3))
        plt.imshow(part_to_motion[None, :], aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.yticks([0], [f"CLS-{PART_NAMES[j]}"])
        plt.xlabel("Time token index")
        plt.title(f"Probe Attention: CLS-{PART_NAMES[j]} -> MotionTokens")
        plt.tight_layout()
        plt.savefig(
            f"{out_path_prefix}_probe_attn_part_{PART_NAMES[j]}.png",
            dpi=150
        )
        plt.close()


# ============================================================
# Qualitative Analysis
# ============================================================

def extract_action_category(text: str) -> str:
    """Extract action category from text"""
    actions = [
        'warm_up', 'walk', 'run', 'jump', 'drink',
        'lift_dumbbell', 'sit', 'eat', 'turn_steering_wheel',
        'phone', 'boxing', 'throw', 'stand', 'turn'
    ]
    t = (text or "").lower()
    for a in actions:
        if a.replace("_", " ") in t or a in t:
            return a
    return "other"


def plot_similarity_heatmap(
    z_t_all: torch.Tensor,
    z_m_all: torch.Tensor,
    epoch: int,
    out_dir: str,
    max_N: int = 256
):
    """Plot text-motion similarity heatmap"""
    ensure_dir(out_dir)
    
    z_t = z_t_all.detach().cpu()
    z_m = z_m_all.detach().cpu()
    N = z_t.size(0)
    
    if N > max_N:
        idxs = torch.linspace(0, N - 1, steps=max_N).long()
        z_t = z_t[idxs]
        z_m = z_m[idxs]
        N = max_N
    
    sims = z_t @ z_m.t()
    
    plt.figure(figsize=(8, 7))
    im = plt.imshow(sims.numpy(), cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="cosine similarity")
    plt.title(f"Text-Motion Similarity Heatmap (Epoch {epoch})")
    plt.xlabel("Motion index")
    plt.ylabel("Text index")
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"1_heatmap_epoch{epoch:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Qualitative] Saved heatmap → {save_path}")


def visualize_query_retrieval(
    z_t_all,
    z_m_all,
    texts,
    ids,
    epoch,
    out_dir,
    num_queries=10,
    top_k=5
):
    """Visualize query-based retrieval results"""
    ensure_dir(out_dir)
    
    z_t = z_t_all.detach().cpu()
    z_m = z_m_all.detach().cpu()
    sim = (z_t @ z_m.t()).numpy()
    N = len(texts)
    
    np.random.seed(epoch + 123)
    query_indices = np.random.choice(N, min(num_queries, N), replace=False)
    
    output_file = os.path.join(out_dir, f"2_query_retrieval_epoch{epoch:03d}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Query-Retrieval Results - Epoch {epoch}\n")
        f.write(f"{'='*80}\n\n")
        
        for qi in query_indices:
            query_text = texts[qi]
            query_id = ids[qi]
            scores = sim[qi]
            top_k_indices = np.argsort(-scores)[:top_k]
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Query #{qi}: {query_text}\n")
            f.write(f"Query ID: {query_id}\n")
            f.write(f"{'='*80}\n")
            
            for rank, idx in enumerate(top_k_indices, 1):
                retrieved_text = texts[idx]
                retrieved_id = ids[idx]
                score = scores[idx]
                is_correct = (idx == qi)
                status = "✓ CORRECT" if is_correct else "✗ Wrong"
                
                f.write(f"\nRank {rank}: {status}\n")
                f.write(f"  ID: {retrieved_id}\n")
                f.write(f"  Text: {retrieved_text}\n")
                f.write(f"  Similarity: {score:.4f}\n")
    
    print(f"[Qualitative] Saved query-retrieval → {output_file}")


def analyze_confusion_by_category(z_t_all, z_m_all, texts, epoch, out_dir):
    """Analyze retrieval accuracy by action category"""
    ensure_dir(out_dir)
    
    z_t = z_t_all.detach().cpu()
    z_m = z_m_all.detach().cpu()
    sim = (z_t @ z_m.t()).numpy()
    N = len(texts)

    categories = [extract_action_category(text) for text in texts]
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    category_ranks = defaultdict(list)

    for i in range(N):
        category = categories[i]
        category_total[category] += 1
        scores = sim[i]
        top1_idx = np.argmax(scores)
        ranking = np.argsort(-scores)
        rank = np.where(ranking == i)[0][0] + 1
        category_ranks[category].append(rank)
        if top1_idx == i:
            category_correct[category] += 1

    output_file = os.path.join(out_dir, f"3_category_analysis_epoch{epoch:03d}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"Retrieval Accuracy by Action Category - Epoch {epoch}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"{'Category':<20} {'Accuracy':<12} {'Count':<10} {'Mean Rank':<12}\n")
        f.write(f"{'-'*70}\n")
        
        for category in sorted(category_total.keys()):
            acc = category_correct[category] / category_total[category]
            count = category_total[category]
            mean_rank = np.mean(category_ranks[category])
            f.write(
                f"{category:<20} {acc:>6.2%}       "
                f"{category_correct[category]:>3}/{count:<3}      "
                f"{mean_rank:>6.2f}\n"
            )
        
        total_acc = sum(category_correct.values()) / max(1, sum(category_total.values()))
        f.write(f"{'-'*70}\n")
        f.write(
            f"{'Overall':<20} {total_acc:>6.2%}       "
            f"{sum(category_correct.values()):>3}/{sum(category_total.values()):<3}\n"
        )
    
    print(f"[Qualitative] Saved category analysis → {output_file}")


def plot_tsne_embeddings(z_t_all, z_m_all, texts, epoch, out_dir, max_samples=300):
    """Plot t-SNE visualization of embeddings"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[WARN] scikit-learn not installed. Skip t-SNE.")
        return

    ensure_dir(out_dir)
    
    z_t = z_t_all.detach().cpu().numpy()
    z_m = z_m_all.detach().cpu().numpy()
    N = len(z_t)

    if N > max_samples:
        indices = np.random.choice(N, max_samples, replace=False)
        z_t = z_t[indices]
        z_m = z_m[indices]
        texts = [texts[i] for i in indices]
        N = max_samples

    z_combined = np.concatenate([z_t, z_m], axis=0)
    print("[Qualitative] Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, N - 1)))
    z_2d = tsne.fit_transform(z_combined)
    
    z_text_2d = z_2d[:N]
    z_motion_2d = z_2d[N:]

    categories = [extract_action_category(text) for text in texts]
    unique_categories = sorted(set(categories))
    category_to_color = {cat: i for i, cat in enumerate(unique_categories)}
    colors = [category_to_color[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        z_motion_2d[:, 0], z_motion_2d[:, 1],
        c=colors, cmap="tab20", alpha=0.6, s=70, marker="o", label="Motion"
    )
    ax.scatter(
        z_text_2d[:, 0], z_text_2d[:, 1],
        c=colors, cmap="tab20", alpha=0.6, s=70, marker="^", label="Text"
    )
    
    for i in range(min(30, N)):
        ax.plot(
            [z_text_2d[i, 0], z_motion_2d[i, 0]],
            [z_text_2d[i, 1], z_motion_2d[i, 1]],
            alpha=0.2, linewidth=0.5
        )

    ax.legend()
    ax.set_title(f"t-SNE: Text and Motion Embeddings (Epoch {epoch})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"4_tsne_epoch{epoch:03d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Qualitative] Saved t-SNE → {save_path}")


def analyze_failure_cases(
    z_t_all,
    z_m_all,
    texts,
    ids,
    epoch,
    out_dir,
    num_failures=20
):
    """Analyze failure cases in retrieval"""
    ensure_dir(out_dir)
    
    z_t = z_t_all.detach().cpu()
    z_m = z_m_all.detach().cpu()
    sim = (z_t @ z_m.t()).numpy()
    N = len(texts)

    failures = []
    for i in range(N):
        scores = sim[i]
        ranking = np.argsort(-scores)
        rank = np.where(ranking == i)[0][0] + 1
        
        if rank > 1:
            top1_idx = ranking[0]
            failures.append({
                "query_idx": i,
                "query_text": texts[i],
                "query_id": ids[i],
                "rank": rank,
                "top1_idx": top1_idx,
                "top1_text": texts[top1_idx],
                "top1_id": ids[top1_idx],
                "correct_score": float(scores[i]),
                "top1_score": float(scores[top1_idx]),
                "score_gap": float(scores[top1_idx] - scores[i]),
            })
    
    failures.sort(key=lambda x: (x["rank"], x["score_gap"]), reverse=True)

    output_file = os.path.join(out_dir, f"6_failure_analysis_epoch{epoch:03d}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Failure Case Analysis - Epoch {epoch}\n")
        f.write(
            f"Total Failures: {len(failures)} / {N} "
            f"({len(failures)/max(1,N)*100:.1f}%)\n"
        )
        f.write(f"{'='*80}\n\n")
        f.write(f"Top {min(num_failures, len(failures))} Worst Failures:\n\n")
        
        for k, fail in enumerate(failures[:num_failures], 1):
            f.write(f"{'='*80}\n")
            f.write(f"[Failure #{k}] Rank: {fail['rank']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Query ID:   {fail['query_id']}\n")
            f.write(f"Query Text: {fail['query_text']}\n\n")
            f.write(f"Retrieved ID:   {fail['top1_id']}\n")
            f.write(f"Retrieved Text: {fail['top1_text']}\n\n")
            f.write(f"Correct Score: {fail['correct_score']:.4f}\n")
            f.write(f"Top-1 Score:   {fail['top1_score']:.4f}\n")
            f.write(f"Score Gap:     {fail['score_gap']:.4f}\n\n")
    
    print(f"[Qualitative] Saved failure analysis → {output_file}")


def comprehensive_qualitative_analysis(
    epoch,
    val_dataset,
    motion_hol_encoder,
    text_encoder,
    tokenizer,
    device,
    out_dir,
    text_max_len=77
):
    """Run comprehensive qualitative analysis"""
    from torch.utils.data import DataLoader
    from dataloader import collate_split_joints
    
    print(f"\n{'='*80}")
    print(f"Running Comprehensive Qualitative Analysis - Epoch {epoch}")
    print(f"{'='*80}\n")

    motion_hol_encoder.eval()
    text_encoder.eval()

    all_z_m = []
    all_z_t = []
    all_texts = []

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_split_joints
    )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting embeddings"):
            hol = batch["holistic"].to(device)
            captions = batch["global_text"]

            tok = tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=text_max_len
            ).to(device)

            z_m_global, _ = motion_hol_encoder(hol)
            z_t = text_encoder(tok)

            all_z_m.append(z_m_global)
            all_z_t.append(z_t)
            all_texts.extend(captions)

    all_z_m = torch.cat(all_z_m, dim=0)
    all_z_t = torch.cat(all_z_t, dim=0)
    all_ids = val_dataset.ids

    epoch_dir = os.path.join(out_dir, f"qualitative_epoch{epoch:03d}")
    ensure_dir(epoch_dir)

    plot_similarity_heatmap(all_z_t, all_z_m, epoch, epoch_dir, max_N=256)
    visualize_query_retrieval(
        all_z_t, all_z_m, all_texts, all_ids, epoch, epoch_dir,
        num_queries=15, top_k=5
    )
    analyze_confusion_by_category(all_z_t, all_z_m, all_texts, epoch, epoch_dir)
    plot_tsne_embeddings(all_z_t, all_z_m, all_texts, epoch, epoch_dir, max_samples=300)
    analyze_failure_cases(
        all_z_t, all_z_m, all_texts, all_ids, epoch, epoch_dir,
        num_failures=30
    )

    print(f"\n[Qualitative] All analyses completed for epoch {epoch}")
    print(f"[Qualitative] Results saved to: {epoch_dir}\n")