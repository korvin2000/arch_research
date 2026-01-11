# Architecture Draft: Efficient Hybrid Restoration + SR (EHR-SR)

## 0) Scope + constraints
**Goal:** A resource-efficient, feature-rich image restoration + super-resolution architecture that combines convolutional efficiency with windowed attention, MoE specialization, and high/low-frequency separation.  
**Constraints:** Favor low memory, scalable attention, re-parameterization for fast inference, and modularity (swap blocks based on task/compute).

This draft is built by selecting **non-conflicting, interoperable ideas** from the repository index and fusing them into a minimal, high-performance hybrid.

---

## 1) Design priorities (priority hierarchy)
1. **Reliability + stability:** consistent tensor shapes, residual pathways, robust layer norm/activation choices.
2. **Efficiency:** windowed attention and SSM for long-range context; re-parameterizable convs for deployment.
3. **Texture/detail recovery:** high/low frequency separation, gated convs, and stochastic texture injection.
4. **Modularity:** swappable expert branches; MoE for specialization; skip connections + MSG down/up paths.

---

## 2) Interoperable ideas harvested from repo (non-conflicting)
**Baseline efficiency + conv gating**
* **NAF-like blocks:** efficient conv-only backbone for restoration (UFPNet / NAFBlock).  
  Use as shallow stem and default fallback for low compute.  
  Sources: `AdaRevD/UFPNet/` (NAF blocks).
* **Gated CNN blocks:** strong local detail and stability (GateRV3, MoSRv2).  
  Use for local texture refinement and fusion.

**Windowed attention + FFN**
* **Swin-style window attention** with shifted windows (CFAT / HAT / MicroSR).  
  Integrate as selective stages in the mid/high-res branch.  
* **GDFN-style FFN / gated FFN** for stable, efficient token mixing (Restormer, ParagonSR).

**Long-range attention / sequence modeling**
* **SSM / Mamba-style selective scan** for global context without full attention (ConvMambaSR).  
  Use at lower resolution to keep compute low.

**High/low frequency separation**
* **Frequency module / FFT branch** (AdaIR / SFHformer) for high-frequency restoration.  
  Lightweight FFT or frequency gating (optional).

**MoE specialization**
* **MoE routing** for expert selection (MoCE-IR, MFGHMoE).  
  Use small expert pool with **token-level routing** for texture vs structure vs denoise.

**MSG down/up paths**
* **Multi-scale grouping (MSG)** (MoESR) with hierarchical down/up.
* Use U-Net style **MSG encoder/decoder** with skip connections.

**Re-parameterization**
* **RepConv / large-kernel partial conv** (PLKSR, SpanC).  
  Use for kernel fusion at deploy time to reduce latency.

**Noise injection / generative texture**
* **Noise injection** per stage (StyleGAN2-like), controlled by learned scalars.  
  Keep optional for texture-heavy tasks (face restoration, aggressive SR).

---

## 3) Proposed architecture: EHR-SR (Efficient Hybrid Restoration + SR)

### 3.1 High-level structure
```
Input
  ↓
Shallow Stem (RepConv + GatedConv)  →  Skip-0
  ↓
MSG Encoder (multi-scale down; NAF/Gated blocks)
  ↓
Core Hybrid Trunk (window attention + SSM + MoE + freq separation)
  ↓
MSG Decoder (multi-scale up; skip connections)
  ↓
Reconstruction Head (PixelShuffle or multi-scale upsample)
  ↓
Output
```

### 3.2 Core Hybrid Trunk
1) **Windowed Attention Blocks (WAB)**  
   - Swin-style local attention with shift + relative bias.  
   - Use as sparse, periodic stages; keep in medium resolution only.  
2) **SSM Global Context Block (SSMB)**  
   - Mamba selective scan on flattened tokens (downsampled).  
   - Efficient long-range context without full attention.  
3) **Frequency Separation Block (FSB)**  
   - FFT / DCT branch to isolate high-frequency residual; gating with conv.  
   - Merge back with learned scaling.
4) **MoE Texture Experts (MoE-TX)**  
   - 2–4 experts specialized for texture / edges / denoise / smooth regions.  
   - Lightweight router with softmax or top-k gating.
5) **Re-parameterizable Conv Mixers (RepMix)**  
   - Partial large kernels or RepConv branches; fuse at inference.

### 3.3 MSG encoder/decoder
**Encoder** uses NAF-style or Gated blocks; **Decoder** mirrors with cross-scale fusion.  
**Skip connections** preserve spatial detail, combined with lightweight attention gates (optional).

---

## 4) Module details (tensor semantics, invariants)
**Shapes:**  
* Input: `B x C x H x W`  
* Downsample levels: `H/2`, `H/4`, `H/8` (configurable)  
* Token attention only applied at `H/4` or lower to control cost.

**Invariants:**  
1. Each block preserves shape unless explicitly down/up-sampling.  
2. Residual paths remain in same dtype/device; no hidden casts.  
3. Routing in MoE is per-token, normalized (sum=1) to preserve energy.  
4. Frequency branch is gated and optional to avoid artifact amplification.

---

## 5) Pseudocode (implementation sketch)
```python
class EHR_SR(nn.Module):
    def __init__(self, cfg):
        self.stem = RepGatedStem(cfg.in_ch, cfg.width)
        self.encoder = MSGEncoder(cfg)
        self.trunk = HybridTrunk(cfg)  # WAB + SSMB + FSB + MoE + RepMix
        self.decoder = MSGDecoder(cfg)
        self.head = ReconHead(cfg)  # PixelShuffle / multi-scale up

    def forward(self, x):
        x0 = self.stem(x)          # B,C,H,W
        skips, x_enc = self.encoder(x0)
        x_trk = self.trunk(x_enc)
        x_dec = self.decoder(x_trk, skips)
        out = self.head(x_dec)
        return out

class HybridTrunk(nn.Module):
    def forward(self, x):
        # downsampled tokens for attention/SSM
        x = WAB(x)                 # window attention
        x = SSMB(x)                # long-range context (Mamba)
        x = FreqSepBlock(x)        # optional FFT/DCT branch
        x = MoE_TextureExperts(x)  # token-level routing
        x = RepMix(x)              # re-parameterizable conv
        return x
```

---

## 6) Suggested implementation anchors in this repo
Use these as reference implementations / code basis:
* **NAF / UFPNet blocks:** `AdaRevD/UFPNet/`
* **Windowed attention:** `CFAT/cfat.py`, `team18_XiaomiMM/model_2.py`, `team07_MicroSR/MicroSR_Model.py`
* **SSM/Mamba:** `ConvMambaSR/ConvMambaSR_arch.py`
* **Frequency separation:** `SFHformer/SFHformer.py`, `SIPL/adair_arch.py`
* **MoE routing:** `MoCE-IR/moce_ir.py`, `neosr/mfghmoe_arch.py`
* **Re-parameterized conv:** `plksr/plksr_arch.py`, `spanpp/spanpp_arch.py`
* **MSG paths:** `moesr/arch.py`
* **Gated CNN:** `gaterv3/gaterv3_arch.py`, `mosrv2/arch.py`

---

## 7) Self-refine (risk checks + mitigations)
**Risk: MoE routing instability**  
Mitigation: add load-balancing loss, entropy regularization, or top-k gating with capacity control.

**Risk: Frequency branch artifacts**  
Mitigation: gated residual branch with learned scalar and clamp or spectral norm.

**Risk: Attention cost at full resolution**  
Mitigation: restrict attention/SSM to downsampled levels only; use window size tuned to GPU memory.

**Risk: Overfitting due to heavy feature fusion**  
Mitigation: light stochastic depth + dropout in FFN/gated blocks; weight decay.

---

## 8) Complexity sketch (qualitative)
* **Baseline:** conv-only (NAF/Gated) at full res — low memory, fast.  
* **Attention:** windowed attention only at lower res — scales with `H*W*ws^2` at reduced resolution.  
* **SSM:** linear in sequence length at low res.  
* **MoE:** per-token top-k expert compute; small expert pool.

---

## 9) Final notes
This design is modular: you can disable MoE, frequency branch, or SSM and still keep a strong baseline. For SR, prefer PixelShuffle head; for restoration, output a residual to add to input. Re-parameterize RepMix before deployment for faster inference.
