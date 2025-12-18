#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py
Motion and text encoder architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel

from MotionDiffuse.models.transformer import MotionTransformer
from utils import PART_NAMES, MOT_PATH


# ============================================================
# Model Loading
# ============================================================

def load_motion_transformer(
    device: torch.device, 
    input_feats: int, 
    max_frames: int
):
    """Load MotionDiffuse transformer backbone"""
    print(f"[INFO] Loading MotionTransformer from: {MOT_PATH}")
    ckpt = torch.load(MOT_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print("[INFO] Detected 'model' key in checkpoint.")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("[INFO] Detected 'state_dict' key in checkpoint.")
    else:
        state_dict = ckpt
        print("[INFO] Treat checkpoint as raw state_dict.")

    latent_dim = 512

    backbone = MotionTransformer(
        input_feats=input_feats,
        num_frames=max_frames,
        latent_dim=latent_dim,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    print(
        f"[INFO] load_state_dict: missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    if missing:
        print("   missing keys (first 10):", missing[:10])
    if unexpected:
        print("   unexpected keys (first 10):", unexpected[:10])

    for _, p in backbone.named_parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )
    print(
        f"[INFO] Motion backbone: total={total_params:,}, "
        f"trainable={trainable_params:,}"
    )

    backbone.train()
    return backbone, latent_dim


def load_text_backbone(device: torch.device):
    """Load CLIP text encoder"""
    from transformers import CLIPTokenizer, CLIPTextModel
    
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)

    text_model.eval()
    for p in text_model.parameters():
        p.requires_grad = False

    hidden_size = text_model.config.hidden_size
    max_ctx_len = text_model.config.max_position_embeddings
    
    print(
        f"[INFO] Loaded CLIP text backbone: {model_name}, "
        f"hidden_size={hidden_size}, max_ctx_len={max_ctx_len}"
    )
    return tokenizer, text_model, hidden_size, max_ctx_len


# ============================================================
# Helper Functions
# ============================================================

def _apply_positional(backbone: MotionTransformer, h: torch.Tensor) -> torch.Tensor:
    """Apply positional encoding (MotionDiffuse version-agnostic)"""
    B, T, D = h.shape
    if hasattr(backbone, "sequence_pos_encoder"):
        return backbone.sequence_pos_encoder(h)
    if hasattr(backbone, "sequence_embedding"):
        seq_emb = backbone.sequence_embedding
        if seq_emb.dim() == 2:
            seq_emb = seq_emb.unsqueeze(0)
        return h + seq_emb[:, :T, :]
    return h


def _run_transformer(backbone: MotionTransformer, x: torch.Tensor) -> torch.Tensor:
    """Run transformer (MotionDiffuse version-agnostic)"""
    if hasattr(backbone, "seqTransEncoder"):
        return backbone.seqTransEncoder(x)
    if hasattr(backbone, "seqTransDecoder"):
        return backbone.seqTransDecoder(x, x)
    return x


# ============================================================
# Attention Probe (for visualization)
# ============================================================

class AttnProbe(nn.Module):
    """
    Attention probe for visualization.
    Separate MHA to extract attention maps without modifying backbone.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.last_attn = None  # (B, heads, L, L)

    def forward(self, x: torch.Tensor):
        out, attn = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn = attn.detach()
        return out

    def get_last(self):
        return self.last_attn


# ============================================================
# Motion Encoders
# ============================================================

class MotionHolisticEncoder(nn.Module):
    """
    Holistic motion encoder.
    
    Architecture:
      motion -> backbone -> [CLS, part-CLS tokens, time tokens] 
      -> global embedding + part embeddings
    
    Optional attention probe for visualization.
    """
    
    def __init__(
        self,
        backbone: MotionTransformer,
        latent_dim: int,
        proj_dim: int,
        num_parts: int,
        use_attn_probe: bool = True,
        probe_heads: int = 4
    ):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.num_parts = num_parts

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.cls_part_tokens = nn.Parameter(torch.randn(1, num_parts, latent_dim))

        # Projection heads
        self.global_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )
        self.part_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

        # Initialize projections
        for m in list(self.global_proj.modules()) + list(self.part_proj.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Attention probe (optional)
        self.attn_probe = (
            AttnProbe(latent_dim, probe_heads) 
            if use_attn_probe else None
        )

    def get_probe_attention(self):
        """Get last attention map from probe"""
        if self.attn_probe is None:
            return None
        return self.attn_probe.get_last()

    def forward(self, motion_hol: torch.Tensor):
        """
        Args:
            motion_hol: (B, T, Dh)
        
        Returns:
            z_global: (B, proj_dim)
            z_parts: (B, num_parts, proj_dim)
        """
        B, T, _ = motion_hol.shape

        if not hasattr(self.backbone, "joint_embed"):
            raise AttributeError(
                "[ERROR] MotionTransformer has no attribute 'joint_embed'. "
                "MotionDiffuse version check required."
            )

        # Embed and add positional encoding
        h = self.backbone.joint_embed(motion_hol)  # (B, T, latent)
        h = _apply_positional(self.backbone, h)

        # Add learnable tokens
        cls = self.cls_token.expand(B, -1, -1)
        cls_parts = self.cls_part_tokens.expand(B, -1, -1)
        x = torch.cat([cls, cls_parts, h], dim=1)  # (B, 1+P+T, latent)

        # Attention probe (for visualization, not in gradient path)
        if self.attn_probe is not None:
            _ = self.attn_probe(x)

        # Transform
        out = _run_transformer(self.backbone, x)

        # Extract and project
        z_global = F.normalize(self.global_proj(out[:, 0, :]), dim=-1)
        z_parts_hidden = out[:, 1 : 1 + self.num_parts, :]  # (B, P, latent)
        z_parts = F.normalize(self.part_proj(z_parts_hidden), dim=-1)  # (B, P, proj)

        return z_global, z_parts


class MotionPartEncoder(nn.Module):
    """
    Part motion encoder.
    
    Architecture:
      part motion (Dp) -> adapter (Dp->Dh) -> backbone -> CLS embedding
    """
    
    def __init__(
        self,
        backbone: MotionTransformer,
        latent_dim: int,
        proj_dim: int,
        in_dim: int,
        target_dim: int
    ):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim

        # Adapter: part dim -> holistic dim
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

        # Initialize
        for m in list(self.adapter.modules()) + list(self.proj.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, motion_part: torch.Tensor):
        """
        Args:
            motion_part: (B, T, Dp)
        
        Returns:
            z_part: (B, proj_dim)
        """
        B, T, _ = motion_part.shape

        # Adapt to holistic dimension
        x = self.adapter(motion_part)  # (B, T, Dh)

        # Embed and add positional encoding
        h = self.backbone.joint_embed(x)  # (B, T, latent)
        h = _apply_positional(self.backbone, h)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, h], dim=1)  # (B, 1+T, latent)

        # Transform
        out = _run_transformer(self.backbone, tok)

        # Project CLS
        z = F.normalize(self.proj(out[:, 0, :]), dim=-1)
        return z


# ============================================================
# Text Encoder
# ============================================================

class TextCLSEncoder(nn.Module):
    """
    Text encoder using CLIP.
    Only the projection head is trainable.
    """
    
    def __init__(
        self,
        backbone: CLIPTextModel,
        hidden_size: int,
        proj_dim: int = 512
    ):
        super().__init__()
        self.backbone = backbone
        
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )
        
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokenized_batch) -> torch.Tensor:
        """
        Args:
            tokenized_batch: CLIP tokenizer output
        
        Returns:
            z: (B, proj_dim) normalized text embeddings
        """
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(**tokenized_batch)
            hidden = out.last_hidden_state
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                cls_vec = out.pooler_output
            else:
                cls_vec = hidden[:, 0, :]
        
        z = self.proj(cls_vec.detach())
        z = F.normalize(z, dim=-1)
        return z

    def train(self, mode=True):
        """Override train to keep backbone in eval mode"""
        self.proj.train(mode)
        self.backbone.eval()
        self.training = mode
        return self