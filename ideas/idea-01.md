# SOTA Image Restoration + Super-Resolution Architecture Draft

## Scope and invariants
- **Tasks**: joint image restoration (denoise/deblur/artifact removal) + SR.
- **Hard constraints**: efficient compute, strong texture recovery, long-range context, frequency separation, MoE specialization, re-parameterizable deployment path.
- **Safety**: preserve input fidelity under mild degradations; avoid texture hallucination unless explicitly gated.

Key invariants:
- Preserve **low-frequency structure** (geometry, luminance) while enhancing **high-frequency textures**.
- Keep **global coherence** without quadratic attention cost.
- Allow **adaptive compute** (MoE / routing) for hard regions only.

---

## Source ideas (non-conflicting, interoperable)
From repository architectures (readme/overview):
- **Re-parameterizable large-kernel convs** for wide receptive fields and deploy speed (PLKSR / SpanC).
- **State-space global context** (ConvMambaSR / Restore_RWKV) for linear-time long-range modeling.
- **Kernel-aware conditioning** for blur handling (UFPNet).
- **Deformable or window attention** for local detail (MSDeformableNAIR / HAT / CFAT / DAT).
- **MoE routing** for specialization and compute efficiency (MoCEIR / MFGHMoE).
- **Frequency-domain mixing** for texture recovery (SFHformer / GFISRV2).
- **Prompted conditioning** for degradation-aware adaptation (PromptIR).
- **GAN-like texture synthesis** with noise injection for fine detail (StyleGAN2).

---

## Four candidate architecture approaches (Tree-of-Thought)

### Approach A — **Hybrid CNN + SSM + Window Attention (No MoE)**
**Core idea**: PLK reparam backbone + SSM global context + window attention for textures.
- **Pros**: Efficient, strong global context, stable.
- **Cons**: Lacks adaptive compute and specialization; harder to cover diverse degradations.

### Approach B — **Transformer-centric (Prompt + Deformable)**
**Core idea**: Prompted Restormer + MS deformable attention + frequency branch.
- **Pros**: Highly adaptive, strong restoration.
- **Cons**: Heavy; memory/latency constraints; less resource-efficient.

### Approach C — **MoE-augmented CNN/SSM Hybrid**
**Core idea**: PLK backbone + MoE blocks for texture/detail experts + SSM for long range.
- **Pros**: Adaptive compute, strong textures, efficient.
- **Cons**: MoE routing complexity; stability risks.

### Approach D — **Hybrid CNN + SSM + Window Attention + MoE + Frequency**
**Core idea**: Combine PLK reparam backbone, SSM global context, windowed attention, MoE experts, and explicit frequency separation.
- **Pros**: Best overall trade-off; covers all requirements (global context, texture detail, adaptive compute).
- **Cons**: Complexity; must control instability and cost.

**Selection (ToT outcome):** **Approach D** is best because it satisfies all hard constraints (efficiency, MoE, long-range, frequency separation, strong textures, reparam) while keeping compute bounded via token gating and MoE routing.

---

## Final Proposed Architecture: **Reparam-SSM-FreqMoE U-Net (RSFMoE-Net)**

### 1) High-level topology (encoder–decoder with skip connections)
```
Input → Shallow Conv Stem
      → Encoder Stage 1..S (PLK-Reparam + SSM + WindowAttn + FreqSep)
      → Bottleneck (SSM + MoE + Frequency Fusion)
      → Decoder Stage S..1 (MSG upsample + WindowAttn + MoE)
      → Reconstruction Head (PixelShuffle/DySample) → Output
```

### 2) Core building blocks

#### A) **Baseline block (fast, reparam, stable)**
**PLK-Reparam Block** (from PLKSR/SpanC):
- Partial Large Kernel Conv + channel mixer + EA gate.
- **Reparameterize at inference** to reduce branches to single conv.
- Invariant: keep receptive field large without attention overhead.

#### B) **Long-range context**
**SSM Global Mixer** (ConvMambaSR / Restore_RWKV style):
- Linear-time sequence mixing per spatial dimension (H and W).
- Gated residual to stabilize (pre-norm, RMSNorm).
- Invariant: global coherence without quadratic attention.

#### C) **Local texture attention**
**Windowed Attention + FFN** (Swin/HAT style):
- Shifted windows to mitigate block artifacts.
- Lightweight FFN with convolutional gating.
- Option: deformable attention at low resolution only.

#### D) **Frequency separation + fusion**
**FreqSep Unit** (SFHformer / GFISRV2):
- Split features into **low-frequency (LF)** and **high-frequency (HF)** via FFT.
- Process HF with texture expert; LF with structure expert.
- Fuse via gated conv and residual.

#### E) **MoE routing**
**MoE Expert Block** (MoCEIR / MFGHMoE):
- Top‑k routing with load-balancing loss.
- Experts: {Detail, Texture, Deblur, Denoise}.
- Router conditioned on degradation prompt + local stats.

#### F) **Degradation conditioning**
**Prompt/Kernel Prior** (PromptIR / UFPNet):
- Prompt generator from global pooled features.
- Kernel prior branch estimates blur kernel (optional).
- Prompts modulate MoE routing + attention scales.

#### G) **MSG down/up paths**
**Multi-Scale Gated (MSG) down/up**
- Down: strided conv + gated conv.
- Up: PixelShuffle or DySample + gated conv.
- Skip connections with feature alignment (1×1 conv).

#### H) **Noise injection (controlled)**
Optional **noise injection** in HF branch only, scaled by router confidence.
- For generative texture, enable during training; disable or clamp in inference.

---

## Detailed module design (shape-aware)

Let input be **(B, C, H, W)**.

### Encoder Stage (per scale)
1. **PLK-Reparam Block**: preserves (B, C, H, W).
2. **SSM Global Mixer**: view as sequence of HW tokens; output same shape.
3. **Window Attention + Gated FFN**: windowed mixing with shift.
4. **FreqSep Unit**:
   - FFT → LF, HF; process separately (HF uses MoE texture expert).
   - iFFT → fused features.
5. **Downsample**: strided conv → (B, 2C, H/2, W/2).

### Bottleneck
- 2–4 blocks of [SSM + MoE + FreqSep], no downsample.
  
### Decoder Stage
1. **Upsample**: PixelShuffle/DySample → (B, C/2, 2H, 2W).
2. **Skip fusion**: concat or add + 1×1 conv.
3. **Window Attention + Gated FFN**.
4. **PLK-Reparam Block**.

### Reconstruction Head
- Conv → PixelShuffle (for SR) or conv output (for restoration).
- Optional **Residual-in-Residual** with input for stability.

---

## Routing / compute strategy
- **MoE only on HF branch** to limit compute.
- **Top‑k = 2** experts; load-balance loss.
- **Gating input**: concat {global prompt, local variance, blur kernel code}.
- **Early-exit** optional: if restoration confidence high, skip deeper decoder blocks.

---

## Training objectives (minimal but effective)
- **L1/L2 reconstruction** (fidelity).
- **Perceptual loss** (LPIPS) for textures.
- **FFT loss** on HF magnitude to stabilize detail.
- **MoE balance loss** to prevent expert collapse.
- **Adversarial loss** (optional, HF-only) if generative textures needed.

---

## Efficiency & deployment notes
- **Reparameterize** all PLK/RepConv blocks at export.
- **Window attention only at mid/low resolutions**.
- **SSM global mixing** runs in linear time; avoid full attention.
- **MoE for HF only** keeps compute bounded.

---

## Risks & mitigations
- **MoE instability** → use entropy regularization + top‑k routing + warmup.
- **FFT ringing** → clamp HF gain, apply smooth gating.
- **Noise hallucination** → enable noise only when prompt indicates severe blur.
- **Custom CUDA (SSM)** → provide fallback to window attention only.

---

## Summary of why RSFMoE-Net is best
This design merges **fast reparameterized CNNs** (PLK/SpanC), **linear-time long‑range context** (SSM), **local window attention**, **frequency separation** (FFT), and **MoE specialization**. It delivers strong texture recovery and adaptive compute while keeping inference efficient and deployable.
