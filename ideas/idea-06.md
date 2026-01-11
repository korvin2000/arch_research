# Architecture Draft: **HARMONIC-RX** (Hybrid Adaptive Restoration & Multi-scale Omni-context Network)

> **Goal:** A single, resource-efficient architecture for **image restoration + super-resolution** that blends strong generative texture recovery with stable, fast inference. This draft fuses **local CNN priors**, **windowed attention**, **state-space long-range mixing**, **frequency separation**, and **sparse MoE** in a **minimal, interoperable** design.

---

## 1) Priority Hierarchy (Design Constraints)
1. **Correctness & Stability:** preserve color/structure; no hallucinated geometry.
2. **Resource Efficiency:** O(HW) or O(HW·logW) attention; avoid full quadratic.
3. **Detail Recovery:** strong texture + micro-structure reconstruction.
4. **Generalization:** handle diverse degradations (blur/noise/compression).
5. **Deployability:** re-parameterizable blocks + static shapes for export.

---

## 2) Invariants & Failure Modes (Mental Model)
**Invariants**
- Preserve low-frequency structure; high-frequency branch only refines.
- All feature maps keep consistent spatial sizes per stage.
- Skip connections always merge same resolution.
- MoE routing must be deterministic at inference (top-1 or top-2).

**Failure modes**
- Over-smoothing if high-frequency branch underweighted.
- Ringing/aliasing from aggressive upsample without pre-filtering.
- MoE expert collapse or imbalance.
- Window attention boundary artifacts (fix with overlap or shift).

---

## 3) Contrastive Reasoning (What to Keep vs Avoid)
**Keep (effective + efficient)**
- Windowed attention + FFN blocks (Swin-family style).
- Re-parameterizable large-kernel conv (deploy-time fusion).
- Frequency split (low/high frequency) for texture specialization.
- MoE for selective compute on texture regions.
- Long-range mixing via SSM / selective scan (linear cost).

**Avoid (inefficient / outdated)**
- Full global attention at high resolution.
- Heavy GAN-only backbones in the main path.
- Multi-stage decoders with deep cascades unless early-exit is used.

---

## 4) Architecture Overview (HARMONIC-RX)
**Core idea:** Two coupled streams:
1) **Base stream (structure & color)** – efficient conv + windowed attention.
2) **Detail stream (texture & edges)** – high-frequency expert MoE + reparam conv.

### High-Level Pipeline
```
Input
  └─ Stem (Conv + PixelUnshuffle for efficiency)
      ├─ Low-Frequency Path (LF): Window Attention + Gated FFN + SSM mixing
      ├─ High-Frequency Path (HF): Frequency split + MoE + Reparam Conv
      └─ Cross-Fusion (gated, residual)
  └─ Multi-Scale U-Net style down/up (MSG)
      ├─ Down: Conv-Gated + Local Attention
      ├─ Bottleneck: SSM + Window Attention
      ├─ Up: Skip fusion + Detail injection
  └─ Reconstruction Head (PixelShuffle + RGB refinement)
Output
```

---

## 5) Modules & Non-Conflicting Ideas (Interoperable)

### 5.1 Baseline Block (Efficient Restoration Core)
**Block name:** `HarmonicBlock`
- **Windowed attention** (shifted windows; optional overlap).
- **Gated FFN** (depthwise conv + pointwise; stable and fast).
- **Conv-gated residual** (lightweight NAF-style gating).
- **SSM mixer** (selective scan) in every N blocks for long-range context.

**Why:** captures local detail with attention and global context without quadratic cost.

### 5.2 Frequency Separation
**Block name:** `FreqSplit`
- Apply Laplacian or Haar wavelet split.
- LF path = global structure refinement.
- HF path = textures/edges enhancement + MoE.

**Why:** avoids high-frequency artifacts and keeps structure stable.

### 5.3 MoE Detail Enhancer
**Block name:** `DetailMoE`
- Experts: {LargeKernelConv, DilatedConv, WindowAttention, SSM-Mixer}.
- Router: lightweight gating on gradient magnitude + local variance.
- Top-1 (training: top-2 with load balancing).

**Why:** allocate compute only where texture complexity is high.

### 5.4 Re-parameterizable Large Kernel
**Block name:** `RepLK`
- Multi-branch (3×3, 5×5, 7×7, 1×k + k×1) with fusion.
- At inference, fuse to single conv for speed.

**Why:** large receptive field without runtime overhead.

### 5.5 MSG Down/Up Paths + Skip Connections
**Block name:** `MSG-UNet`
- Downsample via PixelUnshuffle + 3×3 conv.
- Upsample via PixelShuffle with pre-filter conv.
- Skip fusion uses gated cross-attention to avoid noisy skips.

**Why:** multi-scale detail recovery while preserving structural cues.

---

## 6) Draft Architecture (Stage-by-Stage)
**Stem**
- Conv(3→C), PixelUnshuffle(scale=2), Conv(C→C).

**Stage 1 (Full-res features)**
- `HarmonicBlock` × N1
- `FreqSplit` → LF/HF

**Stage 2 (Downsample)**
- Downsample → `HarmonicBlock` × N2
- HF path uses `DetailMoE` + `RepLK`

**Bottleneck**
- `SSM-Mixer` + Window Attention

**Stage 3 (Upsample)**
- Upsample → `HarmonicBlock` × N3
- Cross-fusion (LF/HF) + gated skip merges

**Reconstruction**
- Conv + PixelShuffle × scale
- RGB refine (3×3 conv + residual)

---

## 7) Pseudo Code (PyTorch-like, shape-aware)
```python
class HarmonicRX(nn.Module):
    def __init__(self, in_ch=3, dim=64, scale=4):
        self.stem = nn.Sequential(
            Conv(in_ch, dim, k=3),
            PixelUnshuffle(2),
            Conv(dim*4, dim, k=3)
        )
        self.stage1 = nn.Sequential(*[HarmonicBlock(dim) for _ in range(N1)])
        self.freq = FreqSplit(mode="haar")  # returns (lf, hf)

        self.down = Downsample(dim)
        self.stage2 = nn.Sequential(*[HarmonicBlock(dim*2) for _ in range(N2)])
        self.detail = DetailMoE(dim*2, experts=[
            RepLK(dim*2), DilatedConv(dim*2),
            WindowAttn(dim*2), SSMixer(dim*2)
        ])

        self.bottleneck = nn.Sequential(
            SSMixer(dim*2), WindowAttn(dim*2), GatedFFN(dim*2)
        )

        self.up = Upsample(dim*2, dim)
        self.stage3 = nn.Sequential(*[HarmonicBlock(dim) for _ in range(N3)])

        self.head = nn.Sequential(
            Conv(dim, dim*scale*scale, k=3),
            PixelShuffle(scale),
            Conv(dim, 3, k=3)
        )

    def forward(self, x):
        x = self.stem(x)                  # [B, C, H/2, W/2]
        x = self.stage1(x)
        lf, hf = self.freq(x)

        x = self.down(lf)
        x = self.stage2(x)
        hf = self.detail(hf)

        x = self.bottleneck(x)
        x = self.up(x)

        x = gated_fuse(x, hf)             # align + gated add
        x = self.stage3(x)
        out = self.head(x)
        return out
```

---

## 8) Suggested Implementation Anchors (from repo codebase)
Use these as **drop-in references** (non-conflicting ideas only):
- Window attention + FFN blocks: **Swin/CFAT/DRCT** style.
- Efficient Gated FFN: **NAFNet / Restormer GDFN** style.
- Long-range linear mixer: **Mamba / RWKV-inspired** blocks.
- MoE routing: **MoCE-IR / MFGHMoE** gating style.
- Re-parameterizable large kernels: **PLKSR / SpanC**.
- Frequency split: **HAIR / SFHformer / GFISRV2** (Fourier/wavelet).
- MSG down/up: **U-Net / Restormer-style enc-dec** with PixelShuffle.

---

## 9) Chain of Verification (Minimal Checks)
1. **Shape check:** every stage preserves alignment for skip fusion.
2. **Determinism:** top-1 routing at inference.
3. **Stability:** clamp frequency fusion weights to avoid ringing.
4. **Speed:** confirm O(HW) blocks dominate; avoid global attention.

---

## 10) Self-Refine Notes (Future iterations)
- If artifacts appear: reduce HF gain or add edge-aware loss.
- If texture weak: allow top-2 MoE routing or increase HF channels.
- If slow: drop attention in Stage 1; keep SSM in bottleneck only.

