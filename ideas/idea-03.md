# SOTA Image Restoration + SR Architecture Draft (SOTA‑lean, resource‑efficient)

## Goal + invariants
**Goal:** single architecture that is strong at restoration + SR, with efficient baseline, generative texture ability, long‑range context, high/low‑frequency separation, MoE specialization, and deploy‑time re‑parameterization.  
**Invariants:** (1) preserve input‑conditioned fidelity while enabling high‑freq synthesis, (2) compute scales sub‑quadratically with image size, (3) deploy path collapses multi‑branch convs, (4) routing is stable (no collapse), (5) frequency path does not amplify ringing.

---

## Base ingredients (from repo evidence)
- **Re‑parameterizable large‑kernel CNN backbone** (PLKSR, SpanC): efficient wide receptive field with deploy‑time kernel fusion.  
- **Kernel‑aware front‑end** (UFPNet): explicit degradation conditioning for blur.  
- **Frequency mixing** (SFHformer / GFISRV2): FFT‑based high/low frequency separation.  
- **Long‑range context** (ConvMambaSR / Restore‑RWKV): state‑space scanning for linear‑time global modeling.  
- **Windowed attention + FFN** (CFAT/HAT/MicroSR): strong local texture modeling with manageable cost.  
- **MoE routing** (MoCEIR / MFGHMoE): expert specialization for degradation types.  
- **Generative texture knobs** (StyleGAN2‑like noise injection, modulated conv ideas): controlled stochastic detail.  
- **Convolutional gating + MSG down/up paths** (MoESR): multiscale gating + MSG path for texture/refinement.  

---

## Four candidate approaches (Tree‑of‑Thoughts, ToT)

### Approach A — **CNN‑first + Frequency + MoE (fastest)**
**Core:** PLKSR‑style large‑kernel blocks + frequency branch (FFT) + MoE experts.  
**Why:** best latency; convs + reparam are GPU‑friendly.  
**Risk:** global coherence weaker unless added long‑range context.

### Approach B — **SSM‑global + CNN‑local + Windowed attention (balanced)**
**Core:** conv stem + SSM (Mamba/RWKV) global stream + window attention for local textures.  
**Why:** linear‑time global context; strong local detail.  
**Risk:** SSM kernel availability; careful fp16 stability.

### Approach C — **Kernel‑aware + Deformable attention + Prompted (robust to blur)**
**Core:** UFPNet kernel conditioning + MS deformable attention + prompt conditioning.  
**Why:** strongest for spatially varying blur/noise.  
**Risk:** heavier compute; more complex training.

### Approach D — **Generative‑lean SR with noise injection + MoE (perceptual)**
**Core:** StyleGAN2‑style noise + modulated conv in texture head; MoE selects texture experts.  
**Why:** best perceptual texture synthesis.  
**Risk:** hallucinations; requires GAN loss; expensive.

---

## ToT Selection (best overall): **Approach B with targeted additions from A/C/D**
**Rationale:**  
1) **Efficiency:** CNN + SSM + windowed attention gives linear‑time global context without full attention cost.  
2) **Texture fidelity:** window attention + frequency branch recovers local detail.  
3) **Robustness:** kernel‑aware front‑end improves deblurring without heavy deformable attention.  
4) **Generative ability:** optional stochastic noise injection only in high‑freq head to avoid hallucinations.  
5) **Deployability:** re‑parameterizable conv branches collapse at inference.

---

## Proposed architecture: **HERMES‑IR/SR (Hybrid Efficient Restoration with MoE + SSM)**

### 0) Input + conditioning
- **Input:** LR/blurred image `x` (B×3×H×W).  
- **Kernel prior head:** shallow CNN predicts per‑pixel kernel features `k` (B×Ck×H×W).  
- **Global degradation token:** pooled summary `g` (B×Cg), used for routing and prompts.

### 1) Dual‑stream stem (local + global)
**Local stream (L):**  
`Conv(3→C) → PLKBlock×N1` with partial large‑kernel convs + gating.  
**Global stream (G):**  
`Conv(3→C) → SSMBlock×N1` (Mamba/RWKV‑style scan) for long‑range context.

### 2) Frequency separation module (F‑split)
- **FFT branch:** `rFFT(x)` → frequency gating → `iFFT` → `F_high`, `F_low`.  
- Inject `F_high` into texture head; `F_low` into structure head.  
- Use lightweight spectral gating to prevent ringing.

### 3) Cross‑stream fusion + window attention
**FusionBlock (repeated N2):**
1. **Cross‑gating:** G modulates L via sigmoid channel gates; L modulates G via depthwise conv gates.  
2. **Windowed attention:** local windows over L with shifted windows every other block.  
3. **FFN:** gated FFN (GDFN‑style) with depthwise conv for local mixing.  

### 4) MoE specialization (texture/structure experts)
- **Router:** `g` + pooled `L` features → top‑k expert selection.  
- **Experts:**  
  - **E1 (Texture)**: conv + window attention + FFT‑aware gating.  
  - **E2 (Structure)**: SSM + low‑freq emphasis.  
  - **E3 (Deblur)**: kernel‑aware conv stack (UFPNet‑style).  
  - **E4 (Generative)**: modulated conv + noise injection (optional).  
- Combine outputs by gated sum (router weights).

### 5) Multi‑scale MSG down/up paths
**MSG block:** downsample L/G → process → upsample, with skip connections at each scale.  
**Reason:** improves context while preserving detail; gating suppresses over‑smoothing.

### 6) Reconstruction head
- **Residual trunk:** re‑parameterizable convs (RepConv/PLK) → fuse at eval.  
- **Upsampler:** PixelShuffle / IGConv for arbitrary scale (optional).  
- **Output:** `y = x_upsampled + residual`.

---

## Design rationale (critical + concise)
- **Baseline efficiency:** PLK/RepConv backbone + reparam yields fast inference.  
- **Global coherence:** SSM stream captures long‑range structure with linear complexity.  
- **High/low frequency control:** FFT split prevents texture hallucination leaking into structure.  
- **MoE specialization:** experts handle distinct degradation modes; router uses global token.  
- **Window attention:** local texture modeling without quadratic cost.  
- **Generative detail (controlled):** noise injection only in expert E4; gate off for PSNR‑focused inference.

---

## Failure modes + safeguards
1) **Routing collapse** → add entropy regularization on router; track expert utilization.  
2) **FFT ringing** → spectral gate + clamp high‑freq gain; include anti‑ringing loss.  
3) **SSM instability** → keep dt/decay in safe ranges; prefer bf16/float32 in SSM path.  
4) **Hallucination** → disable generative expert for fidelity tasks; constrain noise amplitude.  

---

## Minimal training recipe (high level)
- **Losses:** L1 + MS‑SSIM + perceptual; optional GAN loss only if E4 enabled.  
- **Routing:** top‑k (k=2) with load‑balancing loss.  
- **Ablations:** remove SSM / MoE / FFT to validate contributions.  

---

## Pseudocode sketch (shape‑aware)
```
x: (B,3,H,W)
k = KernelPrior(x)                  # (B,Ck,H,W)
g = GlobalPool(x)                   # (B,Cg)
L = PLKStem(x, k)                   # (B,C,H,W)
G = SSMStem(x)                      # (B,C,H,W)
(F_low, F_high) = FFTSplit(x)       # (B,3,H,W) each
L = L + proj(F_high); G = G + proj(F_low)
for i in range(N2):
    L, G = FusionBlock(L, G)
experts = [E1(L,G,k), E2(L,G), E3(L,k), E4(L, noise)]
weights = Router(g, L)
F = sum(w * e for w,e in zip(weights, experts))
F = MSG(F) + L
out = Upsample(RepConv(F)) + Upsample(x)
```

---

## Expected properties
- **Compute:** O(HW) + window attention O(HW·W²) with small window size.  
- **Memory:** linear in HW plus window buffers; no global attention maps.  
- **Quality:** strong detail from FFT + window attention + MoE; global consistency from SSM.
