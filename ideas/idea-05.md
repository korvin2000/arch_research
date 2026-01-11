# Hybrid Restoration + SR Architecture Draft (resource-efficient, feature-rich)

## 0) Scope + priorities (concise)
- **Tasks:** image restoration + super-resolution (single model with scale-aware upsampling).  
- **Hard priorities:** resource efficiency, strong texture/detail recovery, robust to degradations, scalable depth/width.  
- **Key constraints:** avoid full global attention; prefer windowed/hierarchical attention, re-parameterizable convs, and selective routing.

## 1) Non-conflicting ideas harvested from repo (sources)
- **NAF-style efficient CNN baseline** (stable, fast): UFPNet/NAFBlocks. (`AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py`)  
- **Window attention + FFN** (Swin-style): HAT/DRCT/MicroSR/CFAT/HIT-SRF. (`team18_XiaomiMM/model_2.py`, `drct/DRCT_arch.py`, `team07_MicroSR/MicroSR_Model.py`, `CFAT/cfat.py`, `neosr/hitsrf_arch.py`)  
- **Hybrid long-range attention** (efficient, not full global): RHA (FocusedLinearAttention), GRL. (`rha/arch.py`, `grl/grl.py`)  
- **Frequency separation** (Fourier units / frequency modules): SFHformer, GFISRV2. (`SFHformer/SFHformer.py`, `gfisrv2/gfisrv2_arch.py`)  
- **MoE routing** (token/expert efficiency): MoCE-IR, MFGHMoE, MoESR. (`MoCE-IR/moce_ir.py`, `neosr/mfghmoe_arch.py`, `moesr/arch.py`)  
- **MSG down/up paths** (multi-scale guidance): MoESR MSG path. (`moesr/arch.py`)  
- **Re-parameterization + large-kernel conv** (efficient inference): PLKSR / SpanC. (`plksr/plksr_arch.py`, `spanpp/spanpp_arch.py`)  
- **Convolutional gating** (stability & efficiency): GateRV3, MoSRv2. (`gaterv3/gaterv3_arch.py`, `mosrv2/arch.py`)  
- **Generative texture modeling & noise injection**: StyleGAN2 (noise + modulated conv). (`others/stylegan2_arch.py`)

## 1.1 Priority hierarchy (condensed)
1. **Correctness/stability** (no artifacts, predictable behavior).  
2. **Efficiency** (avoid global MHSA; use reparam conv + window attention).  
3. **Texture/detail** (high-frequency branch + MoE + optional noise).  
4. **Scalability** (MSG paths + flexible upsampler).

## 1.2 Reasoning artifacts (concise, non-exhaustive)
- **PoT**: separate low/high frequency, constrain MoE to high branch for stability.  
- **ToT**: evaluated CNN-only vs hybrid vs attention-heavy; chose hybrid for balance.  
- **Contrastive**: avoid full global attention; windowed + lightweight global mixer.  
- **Self-refine**: add reparam and MSG paths for deployment & multi-scale stability.  
- **Verification**: forward-shape + reparam equivalence + MoE routing balance.

## 2) Architecture overview (high-level)
**Name (draft):** *HyReSR-MoE* (Hybrid Restoration + SR with MoE and Frequency-Separated Attention)

**Macro pipeline:**  
Input → **Shallow conv stem** → **Multi-scale encoder (MSG)** → **Hybrid blocks** (window attention + gated CNN + frequency split + MoE) → **Multi-scale decoder (MSG)** → **Reconstruction head** (scale-aware upsampler) → Output

### 2.1 Core design principles
1. **Efficient baseline**: NAF-like gated CNN blocks for stable low-cost restoration.  
2. **High/low frequency split**: explicit frequency branch to prevent over-smoothing and improve texture.  
3. **Selective compute**: MoE routing on mid/high-frequency branch only.  
4. **Long-range context**: window attention + lightweight global mixer (RHA/GRL-like).  
5. **Re-parameterization**: PLK/RepConv reparam for deployment.  
6. **MSG paths**: multi-scale guidance for better structure consistency.  
7. **Noise injection**: optional, for generative texture recovery in high-freq branch.  
8. **Skip connections**: multi-level residual + long skip to stabilize training.

## 3) Module draft (block-level)

### 3.1 HyReSR block (main trunk)
**Input:** `B x C x H x W`

**(A) Frequency splitter**
- `low = AvgPool + depthwise conv`  
- `high = x - Up(low)`  
Optional: FourierUnit to refine both streams (inspired by SFHformer/GFISRV2).

**(B) Low-frequency path (stable baseline)**
- NAF-style gated conv block  
- Optional PLK-style large-kernel reparam branch  
Goal: denoise, deblur, preserve structure.

**(C) High-frequency path (texture branch)**
- Windowed attention (shifted/non-shifted) + Gated FFN  
- MoE experts (2–4) with top-1 routing (tokenwise)  
- Optional noise injection (per-layer, amplitude conditioned on `high` stats).

**(D) Cross-fusion**
- Lightweight cross-attention or concat+1x1 conv to fuse low/high.  
Use channel attention to reweight.

**(E) MSG residual**
- Multi-scale skip aggregation: fuse encoder/decoder features at matched scales (MoESR MSG style).

### 3.2 Hybrid long-range attention (efficient)
Use **windowed attention** for local context + **lightweight global mixer**:
- `WAttn`: windowed MHSA with relative position bias (HAT/DRCT/MicroSR/CFAT).  
- `GlobalMix`: FocusedLinearAttention (RHA) or GRL-style global block at low resolution (downsampled tokens).

This avoids full global MHSA at high resolution while retaining long-range coherence.

### 3.3 Convolutional gating (stability)
Use gated depthwise + pointwise conv blocks (GateRV3/MoSRv2) in the low-frequency path to stabilize and reduce artifacts.

### 3.4 Re-parameterization (deployment)
During training: multi-branch large-kernel conv (PLK/RepConv).  
During inference: fuse branches into a single conv for speed.

## 4) Overall architecture diagram (text)
```
Input
  └─ Shallow Conv Stem
      └─ Encoder (MSG down path, NAF-like)
          └─ [HyReSR Block × L1] (low/high split + MoE + WAttn)
          └─ Downsample (x2)
          └─ [HyReSR Block × L2]
          └─ Downsample (x2)
          └─ Bottleneck [HyReSR Block × L3 + GlobalMix]
      └─ Decoder (MSG up path)
          └─ Upsample (x2) + MSG skip
          └─ [HyReSR Block × L2]
          └─ Upsample (x2) + MSG skip
          └─ [HyReSR Block × L1]
      └─ Reconstruction Head (scale-aware PixelShuffle or DySample)
Output
```

## 5) Pseudocode (minimal, PyTorch-like)
```python
class HyReSR(nn.Module):
    def __init__(self, in_ch=3, base=48, depths=(4, 6, 8), win=8, experts=4, scale=2):
        super().__init__()
        self.stem = Conv3x3(in_ch, base)
        self.enc1 = Stage(base, depth=depths[0], win=win, experts=experts)
        self.down1 = Downsample(base, base*2)
        self.enc2 = Stage(base*2, depth=depths[1], win=win, experts=experts)
        self.down2 = Downsample(base*2, base*4)
        self.mid  = Stage(base*4, depth=depths[2], win=win, experts=experts, use_global_mix=True)
        self.up2  = Upsample(base*4, base*2)
        self.dec2 = Stage(base*2, depth=depths[1], win=win, experts=experts)
        self.up1  = Upsample(base*2, base)
        self.dec1 = Stage(base, depth=depths[0], win=win, experts=experts)
        self.head = ReconHead(base, scale=scale)  # PixelShuffle / DySample

    def forward(self, x):
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        m  = self.mid(self.down2(e2))
        d2 = self.dec2(self.up2(m) + msg(e2))
        d1 = self.dec1(self.up1(d2) + msg(e1))
        return self.head(d1 + x0)

class HyReSRBlock(nn.Module):
    def __init__(self, dim, win, experts, use_global_mix=False):
        super().__init__()
        self.split = FreqSplit(dim)              # low/high
        self.low   = NAFGatedConv(dim)           # stable baseline
        self.high  = WAttnGatedFFN(dim, win)     # windowed attention + FFN
        self.moe   = TokenMoE(dim, experts)      # top-1 routing
        self.fuse  = Conv1x1(dim*2, dim)
        self.mix  = GlobalMix(dim) if use_global_mix else nn.Identity()
        self.reparam = PLKReparam(dim)           # train-time multi-branch, fuse at deploy

    def forward(self, x):
        low, high = self.split(x)
        low = self.low(low)
        high = self.moe(self.high(high))
        z = self.fuse(torch.cat([low, high], dim=1))
        z = self.mix(z)
        return self.reparam(z) + x
```

## 6) Design invariants & failure modes (concise)
- **Invariant:** preserve spatial size across blocks; only down/up in MSG stages.  
- **Invariant:** keep MoE on **high-frequency** branch to avoid instability in low-frequency restoration.  
- **Potential failure:** MoE expert collapse → mitigate with load balancing loss + warmup.  
- **Potential failure:** frequency split artifacts → add learnable fusion weight + soft split.  
- **Potential failure:** window attention boundary artifacts → shift windows + overlap ratio (HAT/DRCT).  
- **Potential failure:** texture hallucination → disable noise injection for non-GAN training.

## 7) Suggested implementations / code references (for reuse)
- **NAF-style baseline / gated conv:** `AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py`  
- **Windowed attention + FFN:** `team18_XiaomiMM/model_2.py`, `drct/DRCT_arch.py`, `team07_MicroSR/MicroSR_Model.py`, `CFAT/cfat.py`  
- **Hybrid/global mixer:** `rha/arch.py`, `grl/grl.py`  
- **Frequency modules:** `SFHformer/SFHformer.py`, `gfisrv2/gfisrv2_arch.py`  
- **MoE routing:** `MoCE-IR/moce_ir.py`, `neosr/mfghmoe_arch.py`, `moesr/arch.py`  
- **MSG down/up path:** `moesr/arch.py`  
- **Re-parameterization:** `plksr/plksr_arch.py`, `spanpp/spanpp_arch.py`  
- **Noise injection (GAN-style):** `others/stylegan2_arch.py`

## 8) Minimal training checklist (verification)
- **Forward shape check:** random input, verify output shape for scales 1/2/4.  
- **Grad check:** ensure MoE routing differentiable (straight-through or Gumbel).  
- **Performance sanity:** profile attention windows vs resolution; ensure no global MHSA.  
- **Reparam check:** fuse PLK/RepConv and verify output equivalence within tolerance.

## 9) Optional variants (resource tuning)
- **Tiny:** remove GlobalMix + reduce experts to 2 + smaller window.  
- **SR-heavy:** add overlap attention + deeper high-frequency branch.  
- **Restoration-heavy:** increase low-frequency depth + reduce MoE usage.
