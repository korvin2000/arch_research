# NEXUS: Neural EXtreme Upscaling System

## A Next-Generation Super-Resolution Transformer Architecture

NEXUS is a state-of-the-art image super-resolution transformer that synthesizes the best innovations from HAT, DAT, and DRCT while introducing novel frequency-aware attention and hardware-optimized design.

---

## 🎯 Design Philosophy

NEXUS was designed collaboratively by analyzing three state-of-the-art architectures:

| Source | Key Innovation Adopted |
|--------|----------------------|
| **HAT** | Hybrid channel attention (CAB) for global feature recalibration |
| **DAT** | Dual spatial-channel aggregation for orthogonal feature processing |
| **DRCT** | Information preservation principles (without memory-heavy dense connections) |
| **Novel** | Frequency-enhanced attention for texture preservation |
| **Novel** | ALiBi position encoding for resolution agnosticism |

---

## 🏗️ Architecture Overview

### Core Components

```
NEXUS Block:
┌─────────────────────────────────────────────────────────────┐
│  Input (B, N, C)                                            │
│     │                                                       │
│     ├──► Efficient Channel Attention (ECA)                  │
│     │         └──► Depthwise separable conv                 │
│     │         └──► Squeeze-excitation                       │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────────────────────────────────────┐           │
│  │  Dual-Dimension Attention (parallel)         │           │
│  │  ├── Spatial Path: Window attention + ALiBi  │           │
│  │  └── Channel Path: Channel self-attention    │           │
│  │           ↓                                  │           │
│  │  Adaptive Fusion (learnable gates)           │           │
│  └──────────────────────────────────────────────┘           │
│     │                                                       │
│     ├──► [Optional] Frequency Enhancement                   │
│     │         └──► High/Low frequency decomposition         │
│     │         └──► Frequency-modulated attention            │
│     │                                                       │
│     ▼                                                       │
│  Spatial-Gated FFN                                          │
│     └──► Depthwise conv gating (from DAT's SGFN)            │
│     │                                                       │
│     ▼                                                       │
│  LayerScale + Residual                                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovations

#### 1. ALiBi Position Encoding (replacing RPB tables)
```python
# Instead of:
relative_position_bias_table = Parameter(torch.zeros((2*window-1)^2, heads))

# We use:
bias = -slope * |query_pos - key_pos|  # Linear distance penalty
```
**Benefits**: Zero parameters, resolution-agnostic, extrapolates to any window size

#### 2. Dual-Dimension Attention
- **Spatial Path**: Standard window attention with shifted windows
- **Channel Path**: Self-attention across feature channels
- **Fusion**: Learnable weighted combination (softmax gating)

#### 3. Frequency-Enhanced Attention
```python
# High-frequency features modulate attention weights
Q_f = Q * (1 + α * freq_features)
K_f = K * (1 + α * freq_features)
attn = softmax(Q_f @ K_f.T / sqrt(d))
```
**Benefits**: Preserves texture details, prevents over-smoothing

#### 4. Weighted Aggregation (replacing dense connections)
```python
# Instead of DRCT's:
x = concat([x1, x2, x3, x4, x5])  # Memory explosion!

# We use:
weights = softmax(learnable_weights)
x = sum(w * out for w, out in zip(weights, outputs))  # Constant memory
```

---

## 📊 Model Variants

| Variant | embed_dim | depths | heads | Params (est.) | Use Case |
|---------|-----------|--------|-------|---------------|----------|
| **nexus_tiny** | 128 | (6,6,6,6) | 4 | ~8M | Fast inference, real-time |
| **nexus_small** | 180 | (6,6,6,6,6,6) | 6 | ~15M | Balanced quality/speed |
| **nexus_medium** | 256 | (6,6,6,6,6,6) | 8 | ~25M | RTX 5090 optimized |
| **nexus_large** | 256 | (6×12) | 8 | ~45M | High quality |
| **nexus_extreme** | 320 | (6×14) | 10 | ~70M | Maximum quality |

---

## 🖥️ RTX 5090 Optimizations

Based on Gemini's hardware analysis:

1. **embed_dim=256**: Aligns with GPU memory/cache lines (multiples of 32)
2. **window_size=16**: Good balance of locality and SM occupancy (256 tokens/window)
3. **num_heads=8 (head_dim=32)**: Matches NVIDIA warp size for efficient reductions
4. **mlp_ratio=3**: Reduced from 4 to afford wider channels
5. **ALiBi**: Zero memory lookup overhead
6. **Weighted aggregation**: Avoids DRCT's memory bandwidth bottleneck

---

## 🔧 Installation

### For traiNNer:
```bash
# Copy files to traiNNer
cp nexus_arch.py traiNNer/archs/nexus_full_arch.py
cp nexus_registry.py traiNNer/archs/nexus_arch.py

# Add to archs/__init__.py:
from .nexus_arch import nexus_tiny, nexus_small, nexus_medium, nexus_large, nexus_extreme
```

### Standalone:
```python
from nexus_arch import NEXUS, nexus_medium

model = nexus_medium(scale=4)
output = model(input_image)
```

---

## 📋 Training Configuration (Recommended)

```yaml
# train_nexus_medium_x4.yml
network_g:
  type: nexus_medium
  scale: 4
  in_chans: 3
  img_size: 64
  window_size: 16
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 256
  num_heads: 8
  mlp_ratio: 3.0
  drop_path_rate: 0.1
  upsampler: pixelshuffle
  resi_connection: 1conv

# Training settings
train:
  optim_g:
    type: AdamW
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  
  scheduler:
    type: CosineAnnealingRestartLR
    periods: [250000, 250000, 250000, 250000]
    restart_weights: [1, 1, 1, 1]
    eta_min: !!float 1e-7

# Losses (as specified in requirements)
train:
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
  
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1.0
      'conv4_4': 1.0
      'conv5_4': 1.0
    perceptual_weight: 1.0
  
  gan_opt:
    type: GANLoss
    gan_type: vanilla
    loss_weight: 0.1
```

---

## 🧪 Regularization Techniques (from ChatGPT's Analysis)

### CutBlur Augmentation
```python
# Randomly mix HR/LR patches during training
if random.random() < 0.5:
    mask = generate_random_mask(h, w)
    lr_up = F.interpolate(lr, scale_factor=scale)
    hr_mixed = hr * mask + lr_up * (1 - mask)
```

### Frequency Dropout
```python
# Randomly zero frequency bands during training
if training:
    fft = torch.fft.rfft2(features)
    mask = generate_band_dropout_mask(fft.shape)
    features = torch.fft.irfft2(fft * mask)
```

---

## 📈 Expected Performance

Based on the architectural analysis:

| Metric | HAT-L | DAT | DRCT-XL | NEXUS-Medium (projected) |
|--------|-------|-----|---------|-------------------------|
| FLOPs (64×64) | ~135G | ~115G | ~190G | ~100G |
| Memory | High | Low | Very High | Low-Medium |
| Params | 40.8M | 14.8M | >50M | ~25M |
| Bottleneck | Unfold | Compute | Concat | Balanced |

---

## 🔬 Technical Details

### Attention Complexity

**Spatial Window Attention**: O(HW × M² × C) where M=window_size
**Channel Attention**: O(HW × C²)
**Combined (Dual)**: O(HW × (M² + C) × C)

For typical configs (M=16, C=256):
- M² = 256, C = 256 → Balanced spatial/channel costs

### Position Encoding

ALiBi with geometric slope sequence:
```python
slopes = [2^(-8/h * i) for i in range(1, h+1)]
# For 8 heads: [0.5, 0.25, 0.125, 0.0625, ...]
```

---

## 📚 References

### Primary Sources (analyzed architectures)
1. **HAT**: "Activating More Pixels in Image Super-Resolution Transformer" (CVPR 2023)
2. **DAT**: "Dual Aggregation Transformer for Image Super-Resolution" (ICCV 2023)
3. **DRCT**: "Saving Image Super-Resolution away from Information Bottleneck" (CVPR 2024)

### Techniques Incorporated
4. **ALiBi**: "Train Short, Test Long" - Press et al.
5. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
6. **LayerScale**: "Going deeper with Image Transformers" (CaiT)
7. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention"
8. **FreqFormer**: "Frequency-aware Transformer for Lightweight Image Super-resolution"
9. **CutBlur**: "Rethinking Data Augmentation for Image Super-resolution"

---

## 🤝 Credits

This architecture was designed through collaborative AI research:
- **Claude (Anthropic)**: Architecture synthesis, implementation, integration
- **ChatGPT (OpenAI)**: Theoretical analysis, novel attention mechanisms, regularization strategies
- **Gemini (Google)**: Hardware optimization, efficiency analysis, quantitative profiling

---

## 📄 License

This architecture is released under the MIT License for research and personal use.
For commercial applications, please ensure compliance with any licenses from the original HAT, DAT, and DRCT papers.
