#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
Main training script for GAR split-joints contrastive learning
"""

import os
import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Local imports
from utils import (
    DATA_ROOT, TEXT_DIR, PART_NAMES,
    set_seed, now_str, ensure_dir, save_json, safe_text,
    load_stats_dict, infer_id_width_from_dir, SPLIT_JOINTS_DIR,
    contrastive_loss, soft_contrastive_loss, compute_retrieval_metrics
)
from dataloader import HumanAct12SplitJointsGARDataset, collate_split_joints
from models import (
    load_motion_transformer, load_text_backbone,
    MotionHolisticEncoder, MotionPartEncoder, TextCLSEncoder
)
from visualization import (
    save_attention_maps_probe, comprehensive_qualitative_analysis
)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def run_eval(
    val_loader,
    motion_hol_encoder,
    text_encoder,
    tokenizer,
    device,
    text_max_len: int
):
    """Run evaluation on validation set"""
    motion_hol_encoder.eval()
    text_encoder.eval()

    all_z_m = []
    all_z_t = []
    all_texts = []

    for batch in tqdm(val_loader, desc="[Eval] Extract"):
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

    # Text->Motion
    R1, R5, R10, MR, MedR = compute_retrieval_metrics(all_z_t, all_z_m)
    # Motion->Text
    R1m, R5m, R10m, MRm, MedRm = compute_retrieval_metrics(all_z_m, all_z_t)

    return {
        "z_m": all_z_m,
        "z_t": all_z_t,
        "texts": all_texts,
        "t2m": (R1, R5, R10, MR, MedR),
        "m2t": (R1m, R5m, R10m, MRm, MedRm),
    }


# ============================================================
# Training
# ============================================================

def train_and_eval(args):
    """Main training and evaluation loop"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    set_seed(args.seed)

    # Setup output directory
    ensure_dir(args.out_dir)
    run_id = now_str()
    run_dir = os.path.join(args.out_dir, f"run_{run_id}")
    ensure_dir(run_dir)
    save_json(vars(args), os.path.join(run_dir, "args.json"))
    print(f"[INFO] run_dir = {run_dir}")

    # ID width
    id_width = args.id_width
    if id_width <= 0:
        id_width = infer_id_width_from_dir(SPLIT_JOINTS_DIR)
    print(f"[INFO] id_width = {id_width} (folder names canonicalized to this width)")

    # Load statistics
    stats = load_stats_dict(DATA_ROOT)

    # Holistic dimension
    D_h = stats["holistic"]["mean"].shape[0]
    max_frames = args.max_frames
    print(f"[INFO] Using holistic dim as backbone input_feats: D_h={D_h}, max_frames={max_frames}")

    # Load motion backbone
    motion_backbone, latent_dim = load_motion_transformer(device, input_feats=D_h, max_frames=max_frames)

    if args.freeze_backbone:
        for p in motion_backbone.parameters():
            p.requires_grad = False
        print("[INFO] backbone frozen (requires_grad=False)")

    # Load text backbone
    tokenizer, text_backbone, hidden_size, text_ctx_len = load_text_backbone(device)
    text_max_len = min(args.text_max_len, text_ctx_len)

    proj_dim = args.proj_dim

    # Build encoders
    motion_hol_encoder = MotionHolisticEncoder(
        backbone=motion_backbone,
        latent_dim=latent_dim,
        proj_dim=proj_dim,
        num_parts=len(PART_NAMES),
        use_attn_probe=args.save_attention,
        probe_heads=args.probe_heads
    ).to(device)

    # Part encoders
    part_encoders = nn.ModuleDict()
    for p in PART_NAMES:
        D_p = stats[p]["mean"].shape[0]
        part_encoders[p] = MotionPartEncoder(
            backbone=motion_backbone,
            latent_dim=latent_dim,
            proj_dim=proj_dim,
            in_dim=D_p,
            target_dim=D_h
        ).to(device)

    # Text encoder
    text_encoder = TextCLSEncoder(
        backbone=text_backbone,
        hidden_size=hidden_size,
        proj_dim=proj_dim
    ).to(device)

    # Optimizer parameter groups
    backbone_params = [p for p in motion_backbone.parameters() if p.requires_grad]

    hol_head_params = []
    for name, p in motion_hol_encoder.named_parameters():
        if not name.startswith("backbone.") and p.requires_grad:
            hol_head_params.append(p)

    part_params = []
    for enc in part_encoders.values():
        for name, p in enc.named_parameters():
            if not name.startswith("backbone.") and p.requires_grad:
                part_params.append(p)

    text_proj_params = [p for p in text_encoder.proj.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": hol_head_params, "lr": args.lr},
            {"params": part_params, "lr": args.lr},
            {"params": text_proj_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Datasets
    train_dataset = HumanAct12SplitJointsGARDataset(
        split="train",
        stats=stats,
        max_frames=max_frames,
        gar_json_path=args.gar_train_json,
        id_width=id_width,
        prefer_txt_split=args.prefer_txt_split,
        require_text=True,
        debug=True
    )
    val_dataset = HumanAct12SplitJointsGARDataset(
        split="val",
        stats=stats,
        max_frames=max_frames,
        gar_json_path=args.gar_val_json,
        id_width=id_width,
        prefer_txt_split=args.prefer_txt_split,
        require_text=True,
        debug=True
    )

    print(f"[DEBUG] train_dataset size = {len(train_dataset)}")
    print(f"[DEBUG] val_dataset size   = {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        raise RuntimeError("[FATAL] train_dataset is empty after filtering. ID matching still broken.")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_split_joints
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_split_joints
    )
    
    print(f"[DEBUG] len(train_loader) = {len(train_loader)}")
    print(f"[DEBUG] len(val_loader)   = {len(val_loader)}")

    # Scheduler
    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # AMP
    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[INFO] AMP enabled: {use_amp}")

    # Resume
    start_epoch = 1
    best_r1 = 0.0
    patience_counter = 0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if "motion_hol_encoder" in ckpt:
            motion_hol_encoder.load_state_dict(ckpt["motion_hol_encoder"], strict=True)
        if "part_encoders" in ckpt:
            part_encoders.load_state_dict(ckpt["part_encoders"], strict=True)
        if "text_encoder" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_r1 = float(ckpt.get("best_r1", ckpt.get("R1", 0.0)))
        patience_counter = int(ckpt.get("patience_counter", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(f"[INFO] Resumed from {args.resume}")
        print(
            f"       start_epoch={start_epoch}, best_r1={best_r1:.4f}, "
            f"patience={patience_counter}, global_step={global_step}"
        )

    # Logging
    log_path = os.path.join(run_dir, "train_log.txt")

    def log_line(s: str):
        print(s)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    log_line(f"[INFO] Train start. run_dir={run_dir}")

    # ===== Training Loop =====
    for epoch in range(start_epoch, args.epochs + 1):
        motion_hol_encoder.train()
        text_encoder.train()
        for p in PART_NAMES:
            part_encoders[p].train()

        running_loss = 0.0
        running_g = 0.0
        running_p = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{args.epochs}")
        for it, batch in enumerate(pbar, start=1):
            hol = batch["holistic"].to(device, non_blocking=True)
            parts = {
                p: batch["parts"][p].to(device, non_blocking=True) 
                for p in PART_NAMES
            }
            global_texts = batch["global_text"]
            part_texts = batch["part_texts"]
            B = hol.size(0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 1) Holistic motion -> global embedding
                z_m_global, _ = motion_hol_encoder(hol)

                # 2) Text global
                tok_g = tokenizer(
                    list(global_texts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=text_max_len
                ).to(device)
                z_t_global = text_encoder(tok_g)

                loss_global = contrastive_loss(z_t_global, z_m_global, tau=args.tau)

                # 3) Part-level loss
                loss_part = torch.tensor(0.0, device=device)

                if args.use_gar:
                    # (A) part text embedding
                    z_t_parts = {}
                    for p in PART_NAMES:
                        texts_p = [safe_text(part_texts[p][i]) for i in range(B)]
                        tok_p = tokenizer(
                            texts_p,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=text_max_len
                        ).to(device)
                        z_t_parts[p] = text_encoder(tok_p)

                    # (B) part motion embedding
                    z_m_parts = {}
                    for p in PART_NAMES:
                        z_m_parts[p] = part_encoders[p](parts[p])

                    # (C) soft contrastive per part
                    num_valid = 0
                    for p in PART_NAMES:
                        part_has_text = any(
                            safe_text(part_texts[p][i]) != "" 
                            for i in range(B)
                        )
                        if part_has_text:
                            loss_part = loss_part + soft_contrastive_loss(
                                z_m_parts[p], z_t_parts[p],
                                tau=args.tau, tau_prime=args.tau_prime
                            )
                            num_valid += 1
                    if num_valid > 0:
                        loss_part = loss_part / num_valid

                loss = loss_global + args.lambda_part * loss_part

                # Gradient accumulation scale
                loss_scaled = loss / max(1, args.grad_accum)

            # Backward
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Step on accumulation boundary
            if it % max(1, args.grad_accum) == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(motion_hol_encoder.parameters()) +
                        list(part_encoders.parameters()) +
                        list(text_encoder.parameters()),
                        max_norm=args.grad_clip
                    )

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += float(loss.item())
            running_g += float(loss_global.item())
            running_p += float(loss_part.item()) if args.use_gar else 0.0
            n_batches += 1

            postfix = {
                "loss": f"{loss.item():.4f}",
                "g": f"{loss_global.item():.4f}"
            }
            if args.use_gar:
                postfix["p"] = f"{loss_part.item():.4f}"
            postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix(postfix)

        # Epoch summary
        avg_loss = running_loss / max(1, n_batches)
        avg_g = running_g / max(1, n_batches)
        avg_p = running_p / max(1, n_batches) if args.use_gar else 0.0
        lr_now = scheduler.get_last_lr()[0]
        
        log_line(
            f"[Epoch {epoch}] loss={avg_loss:.4f} (global={avg_g:.4f}, part={avg_p:.4f}), "
            f"lambda_part={args.lambda_part:.4f}, tau={args.tau:.4f}, lr={lr_now:.6e}, "
            f"step={global_step}, freeze_backbone={args.freeze_backbone}"
        )

        # Attention save
        if args.save_attention and (epoch % args.attn_interval == 0 or epoch == 1):
            attn = motion_hol_encoder.get_probe_attention()
            out_prefix = os.path.join(run_dir, f"epoch{epoch:03d}")
            save_attention_maps_probe(
                attn,
                T=max_frames,
                num_parts=len(PART_NAMES),
                out_path_prefix=out_prefix
            )
            log_line(f"[Attn] Saved probe attention maps: {out_prefix}_probe_attn_*.png")

        # Evaluation
        do_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)
        if do_eval:
            eval_out = run_eval(
                val_loader, motion_hol_encoder, text_encoder,
                tokenizer, device, text_max_len
            )

            (R1, R5, R10, MR, MedR) = eval_out["t2m"]
            (R1m, R5m, R10m, MRm, MedRm) = eval_out["m2t"]

            log_line("\n" + "=" * 70)
            log_line(f"Validation Retrieval - Epoch {epoch}")
            log_line("=" * 70)
            log_line(
                f"[Text → Motion] R@1={R1:.4f} ({R1*100:.2f}%), R@5={R5:.4f}, "
                f"R@10={R10:.4f}, MeanR={MR:.2f}, MedR={MedR:.2f}"
            )
            log_line(
                f"[Motion → Text] R@1={R1m:.4f} ({R1m*100:.2f}%), R@5={R5m:.4f}, "
                f"R@10={R10m:.4f}, MeanR={MRm:.2f}, MedR={MedRm:.2f}"
            )
            log_line("=" * 70 + "\n")

            # Qualitative analysis
            if args.run_qualitative and (
                epoch % args.qualitative_interval == 0 
                or epoch == args.epochs 
                or R1 > best_r1
            ):
                comprehensive_qualitative_analysis(
                    epoch=epoch,
                    val_dataset=val_dataset,
                    motion_hol_encoder=motion_hol_encoder,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    device=device,
                    out_dir=run_dir,
                    text_max_len=text_max_len
                )

            # Save best
            improved = (R1 > best_r1)
            if improved:
                best_r1 = R1
                patience_counter = 0
                save_path = os.path.join(run_dir, f"best_epoch{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "best_r1": best_r1,
                        "patience_counter": patience_counter,
                        "global_step": global_step,
                        "t2m": {"R1": R1, "R5": R5, "R10": R10, "MR": MR, "MedR": MedR},
                        "m2t": {"R1": R1m, "R5": R5m, "R10": R10m, "MR": MRm, "MedR": MedRm},
                        "motion_hol_encoder": motion_hol_encoder.state_dict(),
                        "part_encoders": part_encoders.state_dict(),
                        "text_encoder": text_encoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    save_path
                )
                log_line(
                    f"[INFO] ✓ Saved best model → {save_path} "
                    f"(Text→Motion R@1={R1:.4f})"
                )
            else:
                patience_counter += 1
                log_line(
                    f"[INFO] No improvement ({patience_counter}/{args.early_stop_patience}) "
                    f"best_r1={best_r1:.4f}"
                )

                # Periodic checkpoint
                if args.save_every > 0 and (epoch % args.save_every == 0):
                    save_path = os.path.join(run_dir, f"ckpt_epoch{epoch}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_r1": best_r1,
                            "patience_counter": patience_counter,
                            "global_step": global_step,
                            "motion_hol_encoder": motion_hol_encoder.state_dict(),
                            "part_encoders": part_encoders.state_dict(),
                            "text_encoder": text_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "args": vars(args),
                        },
                        save_path
                    )
                    log_line(f"[INFO] Saved checkpoint → {save_path}")

            # Early stopping
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                log_line(f"\n[INFO] Early stopping triggered at epoch {epoch}")
                log_line(f"[INFO] Best Text→Motion R@1 = {best_r1:.4f}")
                break

    log_line(f"\n{'='*80}")
    log_line(f"[Training Complete] Best Text→Motion R@1 = {best_r1:.4f}")
    log_line(f"{'='*80}\n")


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        "GAR Split-Joints Training (Robust ID matching + per-part mean/std + "
        "holistic/part joint learning + probe attention + AMP + resume)"
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--early_stop_patience", type=int, default=1000)

    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--tau_prime", type=float, default=0.05)
    p.add_argument("--proj_dim", type=int, default=512)

    p.add_argument("--use_gar", action="store_true")
    p.add_argument("--lambda_part", type=float, default=1.0)

    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--run_qualitative", action="store_true")
    p.add_argument("--qualitative_interval", type=int, default=50)

    p.add_argument("--text_max_len", type=int, default=77)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument(
        "--out_dir", type=str,
        default="/mnt/e/research/motion_research/contrastive_ckpts_gar_splitjoints_v3"
    )
    p.add_argument("--max_frames", type=int, default=196)

    p.add_argument("--gar_train_json", type=str, default=os.path.join(TEXT_DIR, "gar_train.json"))
    p.add_argument("--gar_val_json", type=str, default=os.path.join(TEXT_DIR, "gar_val.json"))

    # Attention
    p.add_argument("--save_attention", action="store_true")
    p.add_argument("--attn_interval", type=int, default=10)
    p.add_argument("--probe_heads", type=int, default=4)

    # Training features
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--freeze_backbone", action="store_true")

    # Data
    p.add_argument("--prefer_txt_split", action="store_true")
    p.add_argument("--id_width", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)