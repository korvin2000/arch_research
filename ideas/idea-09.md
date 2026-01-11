# HybridMoE-FreqNet: architecture draft

## High-level Overview
- **Goal:** a stable, resource-efficient hybrid SR/restoration baseline with strong perceptual detail, explicit frequency handling, and adaptive compute.
- **Positioning:** conv-first multi-scale backbone (NAF/PLK lineage) augmented with windowed attention and MoE routing for conditional compute.
- **Target use:** single-image SR + real-world restoration (denoise/deblur) under unified training.

## Architecture Diagram (block-level)

Input
→ Shallow stem (3×3 conv)
→ Frequency Splitter (LF/HF)  
→ **Dual-path MSG Encoder-Decoder**
  - LF path: multi-scale conv+attention blocks
  - HF path: texture-focused conv+gating blocks
  - Cross-path fusion at each scale
→ Bottleneck: Windowed Attention + Selective MoE FFN
→ MSG Decoder with skip connections
→ Reconstruction head (PixelShuffle + residual)
→ Output

## Core Modules

| Module name | Function | Inspiration / lineage | Compute & memory impact |
| --- | --- | --- | --- |
| Shallow Stem | Feature lift (3×3 conv) | RRDB / NAFNet stems | Low |
| Frequency Splitter (FFT or Laplacian) | Explicit LF/HF separation | AdaIR (frequency), SFHformer (Fourier) | Low–Medium |
| LF Block (Conv + WindowAttn + GDFN) | Structure, global consistency | Restormer/CFAT/HAT | Medium |
| HF Block (PLK + Gated CNN) | Texture/detail recovery | PLKSR + GateRV3 | Low–Medium |
| Cross-Freq Fusion | Exchange LF↔HF features | Dual-path SR designs | Low |
| Selective MoE FFN | Conditional compute in bottleneck | MoCE-IR / MFGHMoE | Medium (routing overhead) |
| MSG Down/Up | Multi-scale encoding/decoding | U-Net / Restormer | Medium |
| Reparam Large Kernel | Merge conv branches at inference | PLKSR / SpanC | Low at inference |
| Texture Refinement Head | Local detail polish | DetailRefinerNet | Low |

## Data Flow & Frequency Handling
- **Frequency splitter:** produce LF (blurred) + HF (residual/edges) streams; keep same spatial size for simple fusion.
- **LF path:** multi-scale encoder with window attention for long-range consistency and structure reconstruction.
- **HF path:** lightweight PLK + gated conv blocks emphasizing local textures and high-frequency details.
- **Cross-freq fusion:** at each scale, use 1×1 gated fusion (sigmoid gates) to exchange residuals between LF/HF.
- **Decoder:** concatenate skip features from both paths, then refine with texture head before upsampling.

## Attention & Long-Range Modeling Strategy
- **Windowed attention (shifted windows):** provides efficient long-range modeling without global O(N²).
- **Attention placement:** only in LF path and bottleneck; HF path stays convolutional for speed.
- **FFN:** GDFN-style (depthwise conv + gating) for stable training and good detail retention.

## Efficiency & Reparameterization Strategy
- **MoE gating:** top-1 or top-2 routing in the bottleneck FFN; expert capacity small to avoid memory blow-up.
- **Dynamic conv gating:** per-block channel gate to modulate high-frequency amplification.
- **Reparameterization:** PLK blocks with multi-branch large kernels are fused at inference (single conv).
- **Optional token pruning:** skip attention on low-variance windows to reduce compute (safe for LF path).

## Training Considerations
- **Losses:** L1 + perceptual (VGG) + SSIM; optional GAN for texture-heavy datasets.
- **Curriculum:** start with L1/SSIM; add perceptual, then (optional) adversarial late-stage.
- **Stability:** EMA weights, warmup LR, and drop-path in attention blocks.
- **Degradation mix:** synthetic (bicubic + blur/noise) + real-world degradations; keep LF/HF balance.

## Ablation-Ready Design Choices
- Disable MoE → use dense FFN.
- Remove HF path → LF-only model (restoration focus).
- Remove attention → all-conv (speed baseline).
- Swap frequency splitter (FFT ↔ Laplacian).
- Replace PixelShuffle with DySample for adaptive upsampling.

## Risks & Failure Modes
- **MoE collapse:** expert imbalance; mitigate with load-balancing loss or capacity factor.
- **Over-sharpening:** HF path may hallucinate; tune HF gain and perceptual loss weight.
- **Window boundary artifacts:** mitigated via shifted windows and overlap.
- **Memory spikes:** attention at high resolution; cap window size or add token pruning.

## Why This Architecture Is Competitive
- Explicit LF/HF separation ensures stable structure + sharp textures.
- Windowed attention captures long-range context efficiently.
- MoE adds conditional compute without globally increasing cost.
- Reparameterized large kernels keep inference fast while retaining large receptive fields.
- Multi-scale MSG path gives strong restoration and SR across degradations.
