## High-level Overview – design goals & positioning
**Goal:** A resource-efficient, trainable hybrid SR/restoration baseline that combines a stable CNN backbone with selective attention, explicit frequency separation, and conditional compute. The design targets strong perceptual quality and texture recovery while keeping memory predictable and inference scalable.

**Positioning:** CNN-first backbone (NAF/PLK-style) for stability, with windowed attention for long-range dependency, a dual-path frequency decomposition, and an MoE-based detail branch for compute-aware specialization.

---

## Architecture Diagram (textual / block-level)
```
Input
 ├─ Shallow Stem (3x3 conv + GELU)
 ├─ Frequency Decomposition
 │   ├─ LF path: low-pass pooling/stride-2 conv (multi-scale)
 │   └─ HF path: high-pass (x - blur(x)) + 1x1 lift
 ├─ MSG Encoder (3 scales)
 │   ├─ Scale 1: Conv-Gated Blocks + Local Mixers
 │   ├─ Scale 2: Windowed Attention + FFN blocks
 │   └─ Scale 3: Efficient Long-Range (window attention + optional linear/SSM)
 ├─ Bottleneck Fusion (LF↔HF cross-attn + gated conv)
 ├─ MSG Decoder (mirrored; skip connections from encoder)
 ├─ Detail MoE Refinement (2-of-4 experts, token-wise routing)
 ├─ Reparam Large-Kernel Texture Head
 └─ Upsample (PixelShuffle / MoE-upsample) → Output
```

---

## Core Modules (table)
| Module name | Function | Inspiration / lineage | Compute & memory impact |
| --- | --- | --- | --- |
| Shallow Stem | Early feature lift with minimal artifacts | NAF/Conv SR baselines | Very low |
| Frequency Decomposition (LF/HF) | Explicit low/high-frequency separation | SFHformer/AdaIR-style frequency modeling | Low (blur + residual) |
| Conv-Gated Block (CGB) | Depthwise + pointwise with gating | GateRV3/MoSRv2 | Low–moderate |
| Windowed Attention Block (WAB) | Local attention + FFN | HAT/DRCT/CFAT | Moderate |
| Efficient Long-Range Block (ELRB) | Windowed attention + lightweight linear/SSM | ConvMambaSR/GRL | Moderate |
| Cross-Frequency Fusion (CFF) | LF↔HF exchange with gated conv/CA | RHA/AdaIR | Low |
| Detail MoE Refinement (DMR) | Conditional expert selection for textures | MoCE-IR/SeemoRe/MFGHMoE | Moderate (sparse) |
| Reparam Large-Kernel Texture Head (RLT) | Large-kernel conv fused at inference | PLKSR/Aether/SpanC | Low at inference |
| Upsample Head | PixelShuffle or MoE-upsample | MoESR/MFGHMoE | Low–moderate |

---

## Data Flow & Frequency Handling
**Explicit LF/HF split**
1. **Low-frequency (LF):** Apply learned low-pass (stride-2 conv or blur pooling) per scale to stabilize structure and global tone.
2. **High-frequency (HF):** Residual `HF = x - LP(x)` to isolate edges and textures, then channel-lift to match backbone width.
3. **Dual-path processing:** LF path emphasizes structure (attention/SSM), HF path emphasizes texture (conv-gated + large-kernel).
4. **Cross-Frequency Fusion:** Gated fusion at each scale: `LF' = LF + g(HF→LF)`, `HF' = HF + g(LF→HF)` with light channel attention.

**Multi-scale (MSG) path**
Encoder/decoder with 3 scales (×1, ×1/2, ×1/4) and skip connections for stability and reconstruction fidelity.

---

## Attention & Long-Range Modeling Strategy
**Windowed attention + FFN**
* WAB uses shifted windows to capture local context without quadratic global cost.
* FFN uses depthwise conv + gated MLP to preserve local texture.

**Efficient long-range**
* At the coarsest scale, add ELRB with window attention plus a lightweight linear/SSM mixer (no full global attention).
* This provides global coherence at low spatial cost.

---

## Efficiency & Reparameterization Strategy
* **Reparam large-kernel conv** in RLT: train with multi-branch large-kernel + 3x3, then fuse into single conv at inference.
* **MoE sparse routing:** top-2 experts per token in DMR reduces average compute; experts are lightweight conv-gated stacks.
* **Selective attention:** attention only at mid/low resolutions; high-res relies on conv-gated blocks.
* **Optional activation checkpointing** for attention blocks to reduce memory.

---

## Training Considerations (losses, stability, curriculum hints)
**Losses**
* L1 + Charbonnier for fidelity
* Perceptual (VGG) for texture realism
* Frequency loss (FFT magnitude L1) to reinforce HF recovery
* Optional adversarial loss (lightweight GAN) only for perceptual tuning

**Stability**
* Warm-up with reconstruction losses before enabling adversarial.
* MoE load-balancing loss to avoid expert collapse.
* Stochastic depth only in attention stages (low rate, e.g., 0.05–0.1).

**Curriculum**
* Start with synthetic degradation → fine-tune on real-world mixed degradations.
* Progressive scale training (x2 → x4) while keeping architecture fixed.

---

## Ablation-Ready Design Choices (what can be disabled or swapped)
* **MoE detail head** → replace with single expert for deterministic compute.
* **ELRB** → replace with pure window attention if SSM/linear attention not desired.
* **Frequency split** → disable HF path for pure single-stream baseline.
* **Reparam large-kernel head** → switch to standard 3x3 for low-latency builds.
* **MoE upsample** → standard PixelShuffle.

---

## Risks & Failure Modes
* **MoE collapse:** experts underutilized if routing loss is weak.
* **Over-smoothing:** if LF path dominates HF path, texture may wash out.
* **Boundary artifacts:** window attention + shifts can introduce seams if padding is inconsistent.
* **Instability in GAN fine-tune:** adversarial loss can introduce hallucinations; keep optional.

---

## Why This Architecture Is Competitive
* Combines **stable CNN backbone** with **selective attention**, preserving trainability.
* **Explicit frequency separation** ensures dedicated texture recovery and structural fidelity.
* **Conditional compute (MoE)** focuses capacity on hard regions without uniform cost.
* **Multi-scale MSG** with skips retains detail at all resolutions.
* **Reparameterized large kernels** expand receptive field at training-time while staying fast at inference.
