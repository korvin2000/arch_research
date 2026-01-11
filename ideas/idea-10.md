# Architecture Draft: HydraFusionIR (restoration + SR)

Goal: a feature-rich yet resource-efficient image restoration/SR architecture combining complementary ideas (windowed attention + FFN, MoE, frequency separation, convolutional gating, re-parameterization, skip connections, MSG down/up paths, and optional GAN-texture branch). It is designed to be trainable as a single model while supporting efficient inference via re-parameterization and sparse expert routing.

---

## Priority hierarchy (design constraints)
1. **Correctness + stability:** preserve input/output spatial alignment, avoid aliasing, stable normalization, and deterministic routing during eval.
2. **Efficiency:** minimize global attention, use windowed attention + selective experts, exploit large-kernel re-parameterization.
3. **Texture fidelity:** explicit high/low frequency separation, gated detail pathways, optional noise injection for stochastic texture.
4. **Long-range structure:** efficient long-range attention via shifted windows + low-rank global tokens or SSM-lite.
5. **Modularity:** components are swappable (MoE, attention, upsamplers) without breaking tensor contracts.

---

## Architecture index (modules and inspirations)

**Baseline trunk (efficient restoration):**
- **NAF-style gated conv blocks** for stable baseline restoration (fast, strong local detail).
- **Large-kernel re-parameterized convs** in the trunk to expand receptive field at inference without runtime penalty.

**Attention + FFN blocks (local-global balance):**
- **Windowed attention + Gated FFN** for local structure with manageable compute.
- **Shifted-window or overlap-aware windows** to extend receptive field.
- **Optional global summary tokens** or light SSM path to reinforce long-range structure without full global attention.

**Frequency separation:**
- **Dual-path separation**: low-frequency (LF) path for structure + high-frequency (HF) path for details, fused by adaptive gating.
- **Frequency-domain mixing** in HF path (e.g., FFT-inspired channel mixing) for texture recovery.

**MoE routing:**
- **Sparse MoE blocks** in mid/high stages: expert specialization for textures, denoising, deblurring, or artifact suppression.
- Routing conditioned on degradation cues or task prompt.

**MSG down/up paths:**
- **Multi-scale group (MSG) encoder-decoder** with skip connections and gated fusion at each scale.
- Downsample: stride-2 conv or pixel-unshuffle; upsample: pixelshuffle or DySample-like adaptive upsampler.

**Generative texture head (optional):**
- **GAN-friendly texture refiner** attached to HF path with per-layer noise injection.
- Use only when perceptual fidelity is prioritized; can be disabled in deterministic settings.

---

## High-level dataflow
1. **Input**: `x ∈ R[B,C,H,W]` (C=1/3/4, support grayscale/RGB/RGBA).
2. **Stem**: `Conv3x3` → `Feature F0`.
3. **Frequency split**: `F0 → (F_LF, F_HF)` via learned band-split gate.
4. **MSG encoder** (2–3 scales):
   - Per-scale: `NAF/PLK blocks` → `WindowAttn+GFFN` → `MoE (optional)`.
5. **Bottleneck**:
   - `WindowAttn+GFFN` + `ReparamLargeKernelConv` + `MoE`.
6. **MSG decoder**:
   - Upsample → skip-fuse with encoder → `GatedConv` → `WindowAttn+GFFN`.
7. **HF refiner**:
   - `DetailBlock` stack (deformable or FFT-inspired mixers) + optional noise injection.
8. **Fusion + Output head**:
   - `AdaptiveGate([LF, HF])` → `Conv3x3` → output `y`.

---

## Core blocks (concise specs)

### 1) Frequency Separation Gate (FSG)
- **Purpose:** stable LF/HF decomposition without explicit FFT.
- **Form:** `F_LF = Conv_dw(F0)`; `F_HF = F0 - F_LF`; `Gate = sigmoid(Conv1x1(F0))`; output `(Gate*F_HF, (1-Gate)*F_LF)`.
- **Invariant:** preserve shape + dtype; avoid ringing by keeping LF path smooth.

### 2) Windowed Attention + Gated FFN (WAGFFN)
- **Window size:** 8–12, with shifted windows every other block.
- **FFN:** gated depthwise conv + pointwise mixing (fast, stable).
- **Invariant:** `B,C,H,W` preserved, supports mixed precision.

### 3) MoE Expert Block (MoE-X)
- **Experts:** {texture, denoise, deblur, artifact-suppress}.
- **Routing:** top-2 gating with load-balancing loss.
- **Invariant:** output shape preserved; deterministic in eval.

### 4) Re-parameterized Large Kernel Conv (RepLK)
- **Train:** parallel 3x3/5x5/7x7 branches.
- **Deploy:** fused single kernel (no runtime overhead).

### 5) Convolutional Gating (CG)
- **Form:** `Gate = sigmoid(Conv_dw(F))`; output `F * Gate`.
- **Purpose:** content-adaptive suppression of artifacts.

---

## Baseline configuration (efficient default)
- **Scales:** 3 (H→H/2→H/4).
- **Blocks per stage:** [2, 3, 4] encoder + [4, 3, 2] decoder.
- **Attention:** windowed in stages 2–3 only.
- **MoE:** enabled at bottleneck and stage-3; top-1 routing for speed.
- **Reparam:** enabled for trunk convs.

---

## Enhanced configuration (quality-first)
- **MoE:** enabled at stage-2/3 + HF refiner.
- **Attention:** shifted windows + overlap or hybrid tokens.
- **HF refiner:** add noise injection and perceptual GAN head.

---

## Training notes (interoperable, non-conflicting ideas)
- **Losses:** L1 + SSIM + frequency loss; optional GAN + perceptual (VGG/CLIP).
- **MoE balance:** auxiliary load-balancing to prevent expert collapse.
- **Reparam:** fuse after training for deployment.
- **Degradation conditioning:** optional prompt or kernel prior feature for robustness.

---

## Program of Thought (PoT) summary (non-exhaustive)
- **Invariant focus:** shape preservation, stable LF/HF split, deterministic routing at inference.
- **Complexity bound:** windowed attention keeps `O(HW·w²)`; MoE limits per-token compute.

---

## Tree of Thoughts (ToT) summary (non-exhaustive)
- Branch A: CNN-heavy + reparam for speed.
- Branch B: window attention + MoE for adaptivity.
- Branch C: frequency split + HF refiner for textures.
- Selected: A+B+C to balance efficiency and quality.

---

## Contrastive reasoning (concise)
- **Against full global attention:** too expensive at high resolution.
- **Against pure CNN:** insufficient long-range structure for large artifacts.
- **Hybrid choice:** windowed attention + reparam conv achieves balance.

---

## Chain of Draft (CoD) + Self-Refine (brief)
- Draft v1: strong trunk + MoE.
- Refine: add frequency split + MSG skips.
- Final: enforce reparam + optional GAN path for textures.

---

## Chain of Verification (checks)
1. **Tensor contracts:** all blocks preserve `B,C,H,W`.
2. **Stability:** no NaNs with fp16/bf16; gating bounded in [0,1].
3. **Routing determinism:** fix gates in eval; no stochastic noise.
4. **Deployability:** reparam fusion validated (numerical diff < 1e-4).

---

## Risks & mitigations (self-critical)
- **MoE expert collapse:** mitigate with load-balancing loss + temperature.
- **HF noise artifacts:** enable noise injection only in GAN mode.
- **Window boundary artifacts:** use shifted windows + overlap or padding-crop.

---

## Minimal implementation sketch (module list)
- `StemConv`
- `FreqSplitGate`
- `MSGEncoder(stage_i: NAF/PLK + WAGFFN + MoE-X)`
- `Bottleneck(RepLK + WAGFFN + MoE-X)`
- `MSGDecoder(stage_i: Upsample + SkipFuse + CG + WAGFFN)`
- `HFRefiner(optional, GAN-friendly)`
- `HeadConv`

---

## Expected strengths
- Efficient inference via re-parameterization + sparse routing.
- Strong texture/detail via HF refiner + frequency split.
- Long-range consistency via windowed attention + global summary tokens.

