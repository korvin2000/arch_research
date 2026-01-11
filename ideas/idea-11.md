# Architecture Draft: Hybrid Efficient Restoration & Super-Resolution (HERSR)

## Goal
Design a resource-efficient yet feature-rich **image restoration + SR** architecture that unifies **local texture fidelity**, **long-range consistency**, **generative detail**, and **flexible compute**. The draft integrates **frequency separation**, **windowed attention**, **MoE**, **reparameterized large kernels**, **MSG down/up paths**, **skip connections**, and **noise/texture injection**, while keeping compute predictable.

---

## Priority Hierarchy (Design Constraints)
1. **Correctness & stability:** preserve spatial alignment; avoid checkerboard; stable training (no attention blow-ups).
2. **Efficiency:** windowed attention + partial large-kernel convs; avoid full global attention.
3. **Texture/detail recovery:** frequency-domain branch + generative noise injection.
4. **Long-range dependency:** windowed attention with cross-window exchange (shifted/overlap).
5. **MoE specialization:** dynamic expert routing for degradation-specific processing.

---

## Program of Thought (PoT) – Architecture Skeleton
**Input** `x ∈ R^{B×3×H×W}`
1. **Shallow Stem:** `Conv3×3 → GELU → Conv3×3` (feature lift).
2. **Frequency Split:**
   - `LowFreq`: avg-pool or wavelet LL.
   - `HighFreq`: `x - upsample(low)` or wavelet {LH,HL,HH}.
3. **Dual-Path Encoder (MSG Down):**
   - **Local Path:** reparameterized large-kernel conv blocks (PLK-style) + gated CNN.
   - **Context Path:** windowed attention + Gated FFN (Swin/Restormer-style).
4. **MoE Bottleneck:** token-wise router selects from **K experts**:
   - Conv expert (texture/detail).
   - Attention expert (structure).
   - Frequency expert (FFT/wavelet modulation).
5. **Dual-Path Decoder (MSG Up):**
   - Cross-scale fusion of encoder skips (U-Net style).
   - Frequency-aware fusion (low/high combine).
6. **Texture Generator Head:**
   - Noise injection (style-based) + modulation.
   - Residual add to produce final output.

---

## Tree of Thoughts (ToT) – Candidate Component Mix
**Branch A (Efficient baseline):**
- NAF-style gated conv blocks
- Partial large-kernel (PLK) reparameterized conv
- MSG down/up + U-Net skips

**Branch B (Global consistency):**
- Windowed attention + shifted windows
- Cross-window token mixing

**Branch C (Generative detail):**
- StyleGAN2-like noise injection
- Texture modulation head

**Branch D (Adaptivity):**
- MoE routing by degradation type/region

**Selection (Non-conflicting merge):**
Use **A + B + D** as backbone, **C** only at the final refinement head to avoid instability and overhead.

---

## Contrastive Reasoning (Why this mix)
**CNN-only** → efficient but weak long-range structure.  
**Full attention** → heavy memory/compute.  
**Window attention + large kernels + MoE** gives **local detail + long-range coherence** at **predictable cost**.  
**Noise injection only in the last head** avoids destabilizing mid-layer features.

---

## Draft Architecture (Module-Level)

### 1) Shallow Feature Extractor
```
F0 = Conv3×3(x) → GELU → Conv3×3
```

### 2) Frequency Separation
```
L0 = LowPass(F0)            # avgpool or wavelet LL
H0 = F0 - Upsample(L0)      # or wavelet high bands
```

### 3) MSG Encoder (4 scales)
Each scale has **two parallel paths**:
- **Local path (conv):** PLKBlock + gated conv (efficient texture modeling).
- **Context path (attn):** WindowAttention + GatedFFN (long-range).

**Fusion**: concat → 1×1 conv → residual add.

### 4) MoE Bottleneck
Routing: `softmax(W·gap(F))` for K experts. Each expert is **lightweight**:
- **E1:** depthwise + PLK conv (local texture).
- **E2:** windowed attention (structure).
- **E3:** frequency-conditioned block (FFT modulation).

```
F_bottleneck = Σ_i gate_i · Expert_i(F)
```

### 5) MSG Decoder (4 scales)
Upsample via PixelShuffle/DySample, fuse with encoder skips:
```
F_s = Up(F_{s+1}) + Skip_s
F_s = DualPathBlock(F_s)
```

### 6) Texture/Detail Head
Noise injection and texture modulation only at the final stage:
```
F_t = ModulatedConv(F_final, noise)
out = Conv3×3(F_t) + x (residual)
```

---

## Chain of Draft (CoD) → Self-Refine
**Draft 1:** Full generative noise + MoE everywhere → unstable.  
**Draft 2:** Noise only at the head + lightweight MoE → stable.  
**Final:** Keep MoE in bottleneck, noise in head, conv+attn in body.

---

## Chain of Verification (CoV)
1. **Shape invariants:** encoder/decoder scales preserve `H,W` / power-of-two for MSG.
2. **Stability:** no attention on full resolution; windowed only.
3. **Compute:** MoE only at bottleneck; experts lightweight.
4. **Texture recovery:** high-frequency branch + PLK + noise head.

---

## Implementation Notes (PyTorch)
- Use **reparameterizable convs** for deployment (`switch_to_deploy`).
- Prefer **depthwise separable** convs in PLK blocks for memory.
- Window attention: fixed window size (e.g., 8/16); use shift for coverage.
- Use **layer norm** in attention path, **group norm** in conv path.
- Optional **uncertainty or degradation prompt** injected at bottleneck (e.g., PromptIR style).

---

## Minimal Pseudocode (Tensor Semantics)
```
B,C,H,W = x.shape
F0 = stem(x)
L0 = lowpass(F0)
H0 = F0 - up(L0)

F = DualPathEncoder([F0, L0, H0])
F = MoEBottleneck(F)
F = DualPathDecoder(F, skips)

noise = randn_like(F) * sigma
F = modulated_conv(F, noise)
out = head(F) + x
```

---

## Expected Benefits
- **Efficiency:** local PLK convs + windowed attention.
- **Detail recovery:** explicit high-frequency branch + texture noise injection.
- **Adaptivity:** MoE routing for varying degradations.
- **Stability:** no global attention; noise only at head.

