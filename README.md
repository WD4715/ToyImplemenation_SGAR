# SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval

A PyTorch implementation of contrastive learning for 3D human motion retrieval with part-level structural understanding using GAR (Generative Augmentation with Retrieval).

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Structure](#data-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training Features](#training-features)
- [Visualization](#visualization)
- [Citation](#citation)

##  Overview

SGAR learns joint embeddings between 3D human motion and natural language descriptions by:
- **Holistic-Part Decomposition**: Decomposing motion into holistic and body part representations
- **Multi-level Contrastive Learning**: Learning at both global (holistic) and local (part) levels
- **Structural Understanding**: Using part-specific text descriptions to enhance fine-grained motion understanding
- **Soft Contrastive Loss**: Applying GAR-style soft supervision for part-level alignment

### Key Features

 **Part-based Motion Decomposition**: Split joints into 5 body parts (torso, arms, legs)  
 **Per-stream Normalization**: Independent mean/std statistics for each body part  
 **Dual-level Learning**: Global contrastive + Part soft-contrastive losses  
 **Attention Visualization**: Probe-based attention maps for interpretability  
 **Robust ID Matching**: Canonicalized ID system for seamless data loading  
 **Mixed Precision Training**: AMP support for faster training  

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Motion & Text                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐          ┌─────────────────┐
│ Holistic      │          │ Part Motions    │
│ Motion        │          │ (5 parts)       │
│ (T, D_h)      │          │ (T, D_p)        │
└───────┬───────┘          └────────┬────────┘
        │                           │
        ▼                           ▼
┌───────────────┐          ┌─────────────────┐
│ Motion        │          │ Part Encoders   │
│ Holistic      │          │ (w/ Adapters)   │
│ Encoder       │          │                 │
│ + CLS tokens  │          │ Adapter: Dp→Dh  │
└───────┬───────┘          └────────┬────────┘
        │                           │
        │                           ▼
        │                  ┌─────────────────┐
        │                  │ Part Embeddings │
        │                  │ (B, P, proj)    │
        │                  └────────┬────────┘
        │                           │
        ▼                           │
┌───────────────┐                   │
│ Global Emb    │◄──────────────────┘
│ (B, proj)     │
└───────┬───────┘
        │
        │         ┌─────────────────┐
        │         │ Text Encoder    │
        │         │ (CLIP + Proj)   │
        │         │                 │
        │         │ Global & Part   │
        │         │ Descriptions    │
        │         └────────┬────────┘
        │                  │
        └──────────┬───────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Contrastive Losses  │
        │                      │
        │  L_global (InfoNCE)  │
        │  L_part (Soft)       │
        └──────────────────────┘
```

##  Data Structure

### Motion Data

Motion data is stored in the `data/split_joints/` directory with the following structure:

```
data/
├── split_joints/
│   ├── 00000/
│   │   ├── holistic.npy      # Full body motion (T, J_all, 3)
│   │   ├── right_arm.npy     # Right arm joints (T, J_ra, 3)
│   │   ├── left_arm.npy      # Left arm joints (T, J_la, 3)
│   │   ├── right_leg.npy     # Right leg joints (T, J_rl, 3)
│   │   ├── left_leg.npy      # Left leg joints (T, J_ll, 3)
│   │   └── torso.npy         # Torso joints (T, J_t, 3)
│   ├── 00001/
│   │   └── ...
│   └── ...
├── holistic_Mean.npy         # Holistic motion statistics
├── holistic_Std.npy
├── right_arm_Mean.npy        # Per-part statistics
├── right_arm_Std.npy
├── left_arm_Mean.npy
├── left_arm_Std.npy
├── right_leg_Mean.npy
├── right_leg_Std.npy
├── left_leg_Mean.npy
├── left_leg_Std.npy
├── torso_Mean.npy
├── torso_Std.npy
└── texts/
    ├── gar_train.json        # Training annotations
    └── gar_val.json          # Validation annotations
```

**Note**: Each `.npy` file contains motion data in shape `(T, J, 3)`:
- `T`: Number of frames
- `J`: Number of joints for that part
- `3`: 3D coordinates (x, y, z)

### Text Annotations (GAR Format)

Text annotations are stored in JSON format with both global and part-level descriptions:

```json
{
  "id": "00000",
  "caption": "A person goes four and half steps forward.",
  "part_descriptions": {
    "right_arm": "Moving forward with each step",
    "left_arm": "Hanging loosely at sides",
    "right_leg": "Advancing with each step",
    "left_leg": "Advancing with each step",
    "torso": "Maintaining balance and posture"
  }
}
```

**JSON Structure**:
- `id`: Sample identifier (canonicalized to 5 digits, e.g., "00000")
- `caption`: Global holistic description of the motion
- `part_descriptions`: Dictionary of part-specific descriptions
  - Keys: `"right_arm"`, `"left_arm"`, `"right_leg"`, `"left_leg"`, `"torso"`
  - Values: Text describing the motion of that specific body part

**Training/Validation Split**:
- `gar_train.json`: List of annotated samples for training
- `gar_val.json`: List of annotated samples for validation

### Statistics Files

Per-stream normalization statistics (all in `.npy` format):

```python
# Example: Loading statistics
import numpy as np

# Holistic statistics
holistic_mean = np.load('data/holistic_Mean.npy')  # Shape: (D_h,)
holistic_std = np.load('data/holistic_Std.npy')    # Shape: (D_h,)

# Part statistics (example: right_arm)
right_arm_mean = np.load('data/right_arm_Mean.npy')  # Shape: (D_ra,)
right_arm_std = np.load('data/right_arm_Std.npy')    # Shape: (D_ra,)
```

Each statistic file is a 1D array where the dimension equals `J * 3` (number of joints × 3 coordinates).

##  Installation

### 1. Install Requirements

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install transformers
pip install numpy matplotlib tqdm
pip install scikit-learn  # For t-SNE visualization
```

### 2. Clone MotionDiffuse (Required)

SGAR uses the **MotionDiffuse MotionTransformer backbone**.  
You must clone the MotionDiffuse repository as an external dependency.

```bash
git clone https://github.com/ChenFengYe/MotionDiffuse.git
```

**Important Notes**:
- MotionDiffuse is used **only as a backbone encoder**
- **No diffusion sampling** is performed in this project
- We only use the `MotionTransformer` architecture for motion encoding

### 3. Download MotionDiffuse Checkpoint

Place the pretrained checkpoint at:
```
MotionDiffuse/checkpoints/humanml_trans_dec_512_bert/model000600000.pt
```

You can download it from the [MotionDiffuse repository](https://github.com/ChenFengYe/MotionDiffuse#pretrained-models).

##  Usage

### Basic Training

```bash
python train.py \
    --use_gar \
    --lambda_part 1.0 \
    --batch_size 64 \
    --epochs 1000 \
    --lr 1e-4 \
    --eval_interval 10
```

### Full Training with All Features

```bash
python train.py \
    --use_gar \
    --lambda_part 1.0 \
    --batch_size 64 \
    --epochs 1000 \
    --lr 1e-4 \
    --backbone_lr_mult 0.1 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    --grad_accum 1 \
    --tau 0.07 \
    --tau_prime 0.05 \
    --proj_dim 512 \
    --eval_interval 10 \
    --save_attention \
    --attn_interval 10 \
    --run_qualitative \
    --qualitative_interval 50 \
    --amp \
    --prefer_txt_split \
    --out_dir ./outputs \
    --max_frames 196
```

### Resume Training

```bash
python train.py \
    --resume ./outputs/run_YYYY-MM-DD_HH-MM-SS/best_epoch100.pt \
    --use_gar \
    --lambda_part 1.0 \
    ...
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_gar` | False | Enable part-level GAR loss |
| `--lambda_part` | 1.0 | Weight for part-level loss |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 1e-4 | Base learning rate |
| `--backbone_lr_mult` | 0.1 | LR multiplier for backbone |
| `--tau` | 0.07 | Temperature for InfoNCE |
| `--tau_prime` | 0.05 | Temperature for soft contrastive |
| `--proj_dim` | 512 | Projection dimension |
| `--max_frames` | 196 | Maximum sequence length |
| `--save_attention` | False | Save attention visualizations |
| `--run_qualitative` | False | Run qualitative analysis |
| `--amp` | False | Use mixed precision training |
| `--freeze_backbone` | False | Freeze MotionTransformer backbone |

##  Model Components

### 1. Motion Encoders

#### Holistic Encoder
```python
MotionHolisticEncoder(
    backbone=motion_backbone,      # Shared MotionTransformer
    latent_dim=512,
    proj_dim=512,
    num_parts=5,
    use_attn_probe=True
)
```

**Architecture**:
- Input: Holistic motion `(B, T, D_h)`
- Adds learnable `[CLS]` + 5 part-CLS tokens
- Outputs:
  - Global embedding: `(B, proj_dim)`
  - Part embeddings: `(B, 5, proj_dim)`

#### Part Encoder
```python
MotionPartEncoder(
    backbone=motion_backbone,      # Shared backbone
    latent_dim=512,
    proj_dim=512,
    in_dim=D_part,                # Part-specific dimension
    target_dim=D_h                # Holistic dimension
)
```

**Architecture**:
- Input: Part motion `(B, T, D_p)`
- Adapter: `D_p → D_h` (2-layer MLP)
- Adds learnable `[CLS]` token
- Output: Part embedding `(B, proj_dim)`

### 2. Text Encoder

```python
TextCLSEncoder(
    backbone=clip_text_model,     # Frozen CLIP
    hidden_size=512,
    proj_dim=512
)
```

**Architecture**:
- CLIP text encoder (frozen)
- Learnable projection head
- Output: Text embedding `(B, proj_dim)`

### 3. Loss Functions

#### Global Contrastive Loss (InfoNCE)
```python
L_global = InfoNCE(z_text_global, z_motion_global)
```

#### Part Soft-Contrastive Loss (GAR)
```python
L_part = Σ_p SoftContrastive(z_motion_part_p, z_text_part_p)
```

**Total Loss**:
```python
L_total = L_global + λ_part * L_part
```

##  Training Features

### 1. Gradient Accumulation
```bash
--grad_accum 4  # Accumulate gradients over 4 steps
```

### 2. Mixed Precision (AMP)
```bash
--amp  # Enable automatic mixed precision
```

### 3. Learning Rate Schedule
- Cosine annealing with warmup
- Separate LR for backbone (10× smaller by default)

### 4. Early Stopping
```bash
--early_stop_patience 50  # Stop if no improvement for 50 evaluations
```

### 5. Checkpointing
- **Best model**: Saved when validation R@1 improves
- **Periodic checkpoints**: `--save_every N` to save every N epochs
- **Resume**: `--resume path/to/checkpoint.pt`

##  Visualization

### 1. Attention Maps

When `--save_attention` is enabled:
- **CLS → Motion tokens**: Shows which time steps contribute to global embedding
- **Part-CLS → Motion tokens**: Shows part-specific attention patterns

Output: `epoch{N}_probe_attn_*.png`

### 2. Qualitative Analysis

When `--run_qualitative` is enabled:

#### Similarity Heatmap
- Text-motion similarity matrix visualization
- Output: `1_heatmap_epoch{N}.png`

#### Query Retrieval
- Top-K retrieval results for random queries
- Output: `2_query_retrieval_epoch{N}.txt`

#### Category Analysis
- Per-action accuracy breakdown
- Output: `3_category_analysis_epoch{N}.txt`

#### t-SNE Visualization
- 2D embedding space visualization
- Output: `4_tsne_epoch{N}.png`

#### Failure Analysis
- Detailed analysis of worst failure cases
- Output: `6_failure_analysis_epoch{N}.txt`

##  Evaluation Metrics

The model is evaluated using standard retrieval metrics:

- **R@1**: Recall at rank 1 (% of queries where correct match is ranked 1st)
- **R@5**: Recall at rank 5
- **R@10**: Recall at rank 10
- **MeanR**: Mean rank of correct matches
- **MedR**: Median rank of correct matches

Both **Text → Motion** and **Motion → Text** retrieval are evaluated.

##  Output Structure

```
outputs/
└── run_2025-01-01_12-00-00/
    ├── args.json                          # Training configuration
    ├── train_log.txt                      # Training logs
    ├── best_epoch100.pt                   # Best checkpoint
    ├── ckpt_epoch50.pt                    # Periodic checkpoints
    ├── epoch001_probe_attn_cls.png        # Attention visualizations
    ├── epoch001_probe_attn_part_*.png
    └── qualitative_epoch100/              # Qualitative analysis
        ├── 1_heatmap_epoch100.png
        ├── 2_query_retrieval_epoch100.txt
        ├── 3_category_analysis_epoch100.txt
        ├── 4_tsne_epoch100.png
        └── 6_failure_analysis_epoch100.txt
```

##  ID Canonicalization

The system uses a **canonical ID system** to handle ID format inconsistencies:

- Folder names: `00000`, `00001`, ...
- Text files: `1`, `00001`, `1\n`, ...
- JSON: `"1"`, `"00001"`, `1`, ...

All IDs are canonicalized to **5-digit zero-padded strings** (e.g., `"00001"`).

Set ID width manually:
```bash
--id_width 5  # Default: auto-detect from folder names
```

##  Troubleshooting

### Empty Dataset After Filtering

**Problem**: `train_dataset is empty after filtering`

**Solutions**:
1. Check ID format matching:
   ```python
   # In dataset debug output:
   # - folder ids example: ['00000', '00001', ...]
   # - gar ids example: ['00000', '00001', ...]
   ```
2. Ensure JSON IDs match folder names
3. Use `--prefer_txt_split` if you have train.txt/val.txt
4. Check that all required `.npy` files exist in each folder


## 📚 Code Structure

```
.
├── utils.py              # Utilities, losses, metrics
├── dataloader.py         # Dataset and data loading
├── models.py             # Encoder architectures
├── visualization.py      # Attention and qualitative analysis
└── train.py              # Main training script
```

##  Performance Tips

1. **Use AMP**: `--amp` for ~2× speedup on modern GPUs
2. **Gradient Accumulation**: Increase effective batch size without OOM
3. **Freeze Backbone**: `--freeze_backbone` for faster iterations
4. **Adjust Part Loss Weight**: `--lambda_part 0.5` if part loss dominates

##  Citation

If you use this code, please cite:

```bibtex
@article{sgar2025,
  title={SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```


##  Acknowledgments

- MotionDiffuse for the transformer backbone
- CLIP for text encoding
- GAR for soft contrastive learning methodology
