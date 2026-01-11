# Architecture Draft: HARMONIC-IR/SR

## 0) Intent (priority hierarchy)
1. **Restoration fidelity**: preserve structure + recover textures under diverse degradations.
2. **Efficiency**: windowed attention + re-parameterized large kernels + partial compute (MoE).
3. **Stability**: deterministic baseline path with optional generative texture branch.
4. **Interoperability**: compose non-conflicting ideas from the repo index.

## 0.5) Reasoning protocol (requested)
- **Program of Thought (PoT)**: define invariants, failure hypotheses, and verification gates before module choices.
- **Tree of Thoughts (ToT)**: explore 3 branches (conv‑heavy, attention‑heavy, hybrid) and select hybrid for balance.
- **Contrastive reasoning**: compare LF‑safe vs HF‑aggressive paths to prevent structure drift.
- **Chain of Draft (CoD)**: baseline → add frequency split → add MoE → add generative head.
- **Chain‑of‑Reasoning (CoR)**: every added block must justify (a) texture gain, (b) structure safety, (c) compute cost.
- **Chain of Verification**: sanity/ablation/perf checks (see §15).
- **Self‑critical**: flag over‑complexity risk; keep a pure‑conv fast path (§6).

## 1) Core invariants & failure modes
**Invariants**
- Input/feature tensors remain in **NCHW** (float32/float16) with explicit **scale-aware padding** for windowed attention.
- **Low-frequency (LF)** and **high-frequency (HF)** paths remain separable until fusion, to avoid texture hallucination bleeding into structure.
- **Identity-preserving skip** from input to output (residual/long skip) to protect content when restoration is mild.

**Failure hypotheses**
- Over-smoothing: insufficient HF modeling → fix via dedicated HF branch + noise-injected texture head.
- Ringing/texture drift: aggressive attention without LF constraint → fix via LF reconstruction loss and LF gating.
- MoE collapse: route imbalance → fix via load-balancing loss + top‑k gating.

## 2) Baseline backbone (deterministic path)
**Baseline**: NAF/Restormer-style U‑Net with overlap patch embedding + windowed attention + GDFN/FFN.
- Encoder/decoder with **MSG** down/up paths and **skip connections** (U‑Net style).
- Windowed attention blocks with **shifted windows** and **overlap** (Swin/HAT/MicroSR style).
- **Gated FFN** and **channel attention** for texture emphasis.

**Why it’s compatible**: windowed attention scales with resolution, MSG paths preserve multi-scale structure, and U‑Net skips protect LF structure.

## 3) Frequency decomposition & fusion
**LF branch (structure)**
- Downsampled MSG encoder + large‑kernel **re-parameterizable** convolutions (PLK-style) for global structure.
- Optional **state-space/SSM** path (ConvMambaSR idea) for long-range coherence with low cost.

**HF branch (textures)**
- Edge/gradient aware features (Sobel/learned high-pass) + **texture MoE** blocks.
- **Noise injection** in modulated convs (StyleGAN2 idea) only in HF branch.

**Fusion**
- Cross‑attention **LF↔HF** at each decoder stage with lightweight gating.
- Final **frequency-aware residual**: `y = x + w_lf*LF + w_hf*HF`.

## 4) MoE blocks (token + frequency aware)
- **Token router** from CAMixer/SeemoRe/MoCE‑IR: predicts expert routing per spatial token.
- Experts specialize by frequency and degradation type:
  - Expert‑0: denoise/low‑noise
  - Expert‑1: deblur
  - Expert‑2: compression artifacts
  - Expert‑3: texture hallucination (GAN‑leaning)
- **Top‑k gating** with load‑balancing loss; fallback to deterministic expert for stability.

## 5) Long‑range attention (efficient)
- **Hybrid attention**: local window attention + occasional **dilated window** or **global token** to bridge long‑range dependencies.
- Optionally use **deformable neighborhood attention** in bottleneck to improve alignment on blur.
- Efficient channel attention at each stage to reweight texture‑relevant channels.

## 6) Reparameterization & deployment
- Use **re-parameterizable large kernels** (PLK/SpanC style) inside LF branch.
- Deploy-time **fuse** parallel conv paths into single kernels to cut latency.
- Keep a **pure‑conv fast path** (no attention) for low‑resource inference.

## 7) Windowed attention + FFN block (canonical block)
```
Input (N,C,H,W)
 → LN
 → Window Attention (shifted/non-shifted, relative bias)
 → Residual
 → Gated FFN (GDFN / Conv gating)
 → Residual
```
- Optional **ConvGate** pre/post attention to stabilize textures.

## 8) MSG down/up (multi‑scale group) path
- Each stage: 
  - Down: `Conv → PixelUnshuffle` (for efficiency)
  - Up: `PixelShuffle → Conv`
- Cross‑stage **skip connections** at matching scales.

## 9) Generative texture head (optional)
- **Lightweight GAN head** attached to HF branch only.
- **Per‑layer noise injection** for stochastic detail, gated by **IQA‑style metrics** (noise/sharpness).
- Fusion gate ensures LF structure is preserved (no hallucination on structure).

## 10) Training objectives (multi‑term)
- Reconstruction: `L1 + Charbonnier + SSIM`.
- Frequency: LF/HF losses using Laplacian pyramid or FFT magnitude.
- Perceptual: VGG or LPIPS on HF branch.
- MoE: load balancing + router entropy regularization.
- Optional GAN: hinge/relativistic + feature matching.

## 11) Minimal tensor semantics (shape + perf)
- Window attention on `B,C,H,W` → flatten per window → attention → reshape.
- Memory: use **window size 8–16**, head dim <= 32.
- Avoid global attention except bottleneck token or pooled global token.

## 12) Compatibility map (from repo index)
- **Window attention + FFN**: CFAT, MicroSR, HAT families.
- **MoE**: MoCE‑IR, SeemoRe, MFGHMoE.
- **Re‑param large kernels**: PLKSR, SpanC.
- **Noise injection**: StyleGAN2.
- **MSG down/up + U‑Net**: Restormer/PromptIR/NAF‑style models.
- **Hybrid attention**: RHA, DAT.

## 13) Draft module list (code basis)
1. `InputEmbed` (overlap patch embed conv)
2. `StageEnc[i]`:
   - `WindowAttnBlock × N` + `ConvGate`
   - `Downsample` (MSG)
3. `Bottleneck`:
   - `HybridAttn` + `SSM/DeformableAttn`
4. `StageDec[i]`:
   - `Upsample` (MSG) + `SkipFusion`
   - `WindowAttnBlock × N`
5. `LFBranch`:
   - `RepLargeKernelBlocks × N`
6. `HFBranch`:
   - `MoETextureBlocks × N` + noise injection
7. `FreqFusion` + `OutputHead`

## 14) Self‑refine checklist
- **LF integrity**: verify LF branch dominates structure (PSNR on downsampled outputs).
- **HF realism**: evaluate HF branch with perceptual loss + patch‑level FID.
- **MoE balance**: check token/expert utilization and entropy.
- **Ablations**: remove HF branch, remove MoE, remove noise injection.

## 15) Verification plan
- Sanity: identity mapping on clean inputs.
- Stress: blur + noise + compression mix.
- Efficiency: FLOPs/params vs baseline (no attention path).

---
**Short name**: **HARMONIC** (Hybrid Attention + Re‑param MoE + Frequency‑separated IR/SR)
