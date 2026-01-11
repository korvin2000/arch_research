# Hybrid Frequency-MoE SR/IR Backbone (HF-MoE-IR)

## High-level Overview – design goals & positioning
HF-MoE-IR is a resource-efficient, trainable baseline for joint image restoration and super-resolution. It combines a stable SR backbone (windowed-attention + FFN blocks) with explicit frequency separation, conditional computation (MoE), and multi-scale skip paths. The design targets strong perceptual quality and texture recovery while maintaining predictable compute and inference stability.

## Architecture Diagram (textual / block-level)
```
Input LR/Degraded
  └─ Shallow Conv (3x3) + Norm
      └─ Frequency Split:
          ├─ Low-Frequency (LF): depthwise low-pass + downsample
          │    └─ MSG Encoder (2-3 scales)
          │         └─ [Hybrid Windowed Attn Block + MoE FFN] x N
          │         └─ Down/Up with skip connections
          └─ High-Frequency (HF): residual high-pass = x - upsample(LF)
               └─ Texture Branch: dynamic conv-gated blocks + local attention
      └─ Cross-Frequency Fusion (gated cross-attn + concat + 1x1)
  └─ Reconstruction Head:
      ├─ Upsampler (pixel shuffle/unshuffle as needed)
      └─ RepConv (train-time multi-branch → inference fused)
Output SR/Restored
```

## Core Modules (table)
| Module name | Function | Inspiration / lineage | Compute & memory impact |
|---|---|---|---|
| Shallow Conv Stem | Stable feature lift, 3x3 conv + LN/GRN | SwinIR/HAT-style stems; DSwinIR backbone framing | Low |
| Frequency Split (LF/HF) | Explicit low/high separation via depthwise low-pass + residual high-pass | Frequency-aware SR/IR practices; wavelet-like split | Low |
| MSG Encoder-Decoder | Multi-scale down/up with skip connections for robustness | U-Net SR/IR designs | Medium |
| Hybrid Windowed Attn Block (HWAB) | Windowed attention + FFN for long-range dependencies | DSwinIR’s windowed attention emphasis | Medium |
| MoE FFN (Complexity Experts) | Conditional computation with 2–4 experts (cheap→heavy) | MoCE-IR complexity experts | Medium (dynamic) |
| Conv-Gated Texture Block (CGTB) | Local detail recovery with depthwise conv gating | NAFNet/Restormer gating patterns | Low |
| Dynamic Filter Head (DFH) | Predicts per-pixel lightweight kernels for HF branch | Dynamic filtering SR literature | Medium |
| Cross-Frequency Fusion (CFF) | Gated cross-attn + channel mixing | Hybrid IR/SR designs | Medium |
| RepConv Head | Multi-branch conv reparameterized at inference | RepVGG-style reparam | Low (fused) |

## Data Flow & Frequency Handling
- **LF path** uses low-pass filtering + downsampled features for global structure and long-range dependencies. Windowed attention operates only on LF to cap quadratic cost.
- **HF path** uses residual high-pass features for textures and edges; local convolutions + dynamic filtering enhance details.
- **Fusion** uses gated cross-attention from HF→LF and LF→HF, then concatenation and 1x1 projection. This preserves structure while injecting texture.

## Attention & Long-Range Modeling Strategy
- **Windowed attention** with shifted windows on LF scales; optional deformable offsets at the coarsest scale for adaptive receptive fields.
- **Sparse global tokens** (one per window group) aggregate global context without full attention.
- **Cross-frequency attention** uses lightweight keys/values (reduced channels) to limit memory.

## Efficiency & Reparameterization Strategy
- **Conditional MoE**: top-1 or top-2 expert routing with a bias toward low-compute experts; load-balancing loss prevents collapse.
- **Inference fusion**: RepConv (3x3 + 1x1 + identity) collapsed; depthwise + pointwise convs fused where possible.
- **Selective attention**: attention only in LF path; HF path purely convolutional.

## Training Considerations (losses, stability, curriculum hints)
- **Losses**: L1 + frequency loss (FFT/Laplacian), perceptual (VGG), and optional adversarial loss for texture-heavy SR.
- **Stability**: use residual scaling (0.1–0.2), gradient clipping, and cosine LR schedule.
- **Curriculum**: start with synthetic degradations; gradually mix real-world degradations and heavier blur/noise.
- **Optional**: stochastic depth for deep stacks; low-level noise injection in HF branch only.

## Ablation-Ready Design Choices (what can be disabled or swapped)
- Remove MoE routing → single shared FFN.
- Disable dynamic filtering → replace DFH with depthwise conv.
- Drop cross-frequency attention → simple concat + 1x1.
- Use fixed window sizes → no deformable offsets.
- Remove adversarial loss → purely PSNR-optimized baseline.

## Risks & Failure Modes
- **Routing collapse** in MoE: mitigated by load-balancing and temperature annealing.
- **Over-sharpening** from HF branch: control via HF loss weight and gating.
- **Artifact amplification** under real noise: use robust loss (Charbonnier) and HF branch noise injection.
- **Memory spikes** from attention at high resolution: keep attention LF-only and use window size caps.

## Why This Architecture Is Competitive
- Explicit frequency handling aligns with SR/IR physics and improves texture fidelity.
- MoE provides compute-aware scaling without sacrificing performance.
- Windowed attention in LF path gives long-range modeling at controllable cost.
- Reparameterization ensures fast, stable inference.
- Modular design enables clean ablations and targeted improvements.
