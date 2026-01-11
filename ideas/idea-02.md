# Architecture Draft: SOTA‑leaning, Efficient Image Restoration + SR

## Scope + invariants
- **Goals:** strong texture/detail recovery, long‑range context, efficient compute, adaptable to multiple degradations.
- **Hard constraints:** explicit high/low‑frequency separation, MoE blocks, re‑parameterization option, long‑range attention, generative detail support, skip connections, windowed attention + FFN blocks.
- **Operational invariants:** 
  - preserve low‑frequency structure (avoid hallucination),
  - enhance high‑frequency detail (texture),
  - keep compute bounded (routing + windowed ops),
  - stable training (avoid unstable routing + global attention blow‑up).

## Ingredients pulled from repo (non‑conflicting)
- **Efficient baseline blocks:** NAF‑style conv blocks (UFPNet / AdaRevID) + large‑kernel re‑param convs (PLKSR). 
- **Long‑range context:** SSM (ConvMambaSR / Restore‑RWKV) or hierarchical window attention (CFAT/HAT/HIT‑SRF).
- **Frequency separation:** Fourier units (SFHformer / GFISRV2).
- **MoE routing:** MoCEIR / MFGHMoE (top‑k expert routing).
- **Generative texture cues:** StyleGAN2 noise injection / modulation (others/stylegan2).
- **Windowed attention + FFN:** HAT/HIT‑SRF/MicroSR style.
- **Convolutional gating:** GateRV3 / MoSRv2 / RHA.

---

## Four candidate approaches (Tree‑of‑Thoughts)

### Approach A — **Conv‑SSM + Fourier + MoE + Reparam**
**Core idea:** NAF/PLK baseline with optional large‑kernel re‑param branches; mix SSM (Mamba/RWKV) for global context and Fourier units for frequency separation; MoE in mid/high‑res stages.
- **Pros:** strong global context without quadratic attention, frequency separation, efficient deployment via re‑param.
- **Risks:** dependency on custom SSM kernels; MoE routing complexity.

### Approach B — **Window‑Attention + MoE + Frequency split**
**Core idea:** HAT‑style window attention blocks + FFN, with MoE in attention or FFN, and explicit FFT branch for high‑freq refinement.
- **Pros:** stable training, strong detail, proven SR quality.
- **Risks:** higher compute/memory on large inputs vs SSM; attention still heavy.

### Approach C — **Kernel‑aware encoder + deformable attention + MoE**
**Core idea:** UFPNet kernel‑aware encoder, MS‑Deformable attention in bottleneck, MoE in decoder, with PLK re‑param convs in shallow layers.
- **Pros:** robust to blur/motion; adaptive sampling; efficient early layers.
- **Risks:** deformable attention is heavy and can be unstable; complex to train.

### Approach D — **CNN baseline + generative texture injector**
**Core idea:** PLK/NAF backbone for fidelity + StyleGAN2‑like noise injection + lightweight attention for details.
- **Pros:** very strong textures; GAN‑style realism.
- **Risks:** hallucination risk; perceptual vs distortion trade‑off.

---

## Selection (ToT → best candidate)
**Chosen:** **Approach A** — Conv‑SSM + Fourier + MoE + Reparam  
**Reasoning:** 
- Delivers long‑range context at linear cost (SSM), avoids heavy global attention.
- Keeps strong local texture via large‑kernel re‑param convs and FFT split.
- MoE provides specialization without full‑branch cost.
- Retains deterministic, fast CNN backbone.

---

## Final Architecture Draft (SOTA‑leaning + efficient)

### 1) High‑level pipeline
```
Input
  └─ Shallow Stem (Conv + PLK re‑param block)
       └─ Dual‑Path Encoder
           ├─ Low‑freq path: Conv/NAF blocks + downsample
           ├─ High‑freq path: FFT branch (Fourier Unit + gated conv)
       └─ Core Trunk (N stages)
           ├─ Windowed Attention + FFN block (local detail)
           ├─ SSM block (global context, linear cost)
           ├─ MoE Expert Mixer (top‑k routing)
           └─ Fusion + gated residual
       └─ Decoder / Upsampler
           ├─ Skip connections from encoder (multi‑scale)
           ├─ MSG down/up paths (multi‑scale gating)
           ├─ Re‑param PLK refinement blocks
       └─ Texture‑Refine Head
           ├─ Noise injection (optional, low‑amplitude)
           └─ Output conv + residual add
```

### 2) Core blocks (tensor‑semantics aware)
- **PLK‑NAF block (efficient baseline):** 
  - Depthwise + pointwise conv, channel mixing, optional large‑kernel partial conv.
  - **Re‑param**: multi‑branch conv collapsed to single kernel for inference.
- **Frequency split block:**
  - `FFT(x)` → high‑freq branch (gated conv + FFN)  
  - `IFFT` + low‑freq path (conv) → recombine with learned gates.
- **SSM global block:**
  - SSM scan on flattened spatial tokens; ensure fp16/bf16 stability by clamping dt/decay.
- **Windowed attention + FFN:**
  - Local windows with relative position bias; FFN with gated conv.
- **MoE routing block:**
  - Top‑k routing over experts (k=2), load‑balance loss optional.
  - Experts: (a) PLK‑conv expert, (b) FFT‑detail expert, (c) window‑attention expert.
- **MSG down/up paths:**
  - Multi‑scale group paths (low/mid/high resolution) with gated skip fusion.

### 3) Design invariants (correctness + stability)
- **Low‑freq preservation:** residual path always includes unmodified low‑freq features.
- **High‑freq enhancement:** FFT branch + texture head only modulate residual, not base structure.
- **Compute bound:** window attention is local; MoE is top‑k; SSM is linear.
- **Re‑param safety:** conversion step is deterministic and preserves outputs (within FP tolerance).

### 4) Complexity notes
- **Per‑stage cost:**  
  - Window attention: O(HW·w²·C)  
  - SSM scan: O(HW·C)  
  - MoE: O(k·expert_cost) per token  
  - FFT: O(HW log(HW))
- **Bottleneck:** window attention at high resolution; mitigate by placing it after downsampling or limiting stages.

---

## Minimal training recipe (robust + efficient)
- **Loss:** L1 + LPIPS + frequency loss (FFT magnitude) + optional adversarial (low weight).
- **Routing stability:** auxiliary load‑balancing loss; freeze routing for first N epochs.
- **Re‑param deploy:** train with multi‑branch PLK, convert before inference.

---

## Risks + mitigations
- **SSM kernel availability:** provide fallback to window attention‑only trunk.
- **MoE imbalance:** enforce capacity + load‑balancing loss.
- **GAN hallucination:** keep noise injection low; use distortion loss to preserve fidelity.
