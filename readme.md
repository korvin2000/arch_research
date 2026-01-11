# Architecture Index

This repository aggregates PyTorch architectures for **super-resolution (SR)** and **image restoration** (denoising, deblurring, enhancement, hyperspectral restoration, etc.). The table below indexes **66 architecture families** found across the codebase. Variants (when present) are listed in the notes so that every model implementation is represented.

| # | Architecture | Task(s) | Key code signals | Primary location(s) |
| --- | --- | --- | --- | --- |
| 1 | **AdaRevID** | Restoration / deblurring | `NAFBlock`, `Fusion_Decoder` | `AdaRevD/AdaRevID_arch.py` (`AdaRevID`) |
| 2 | **UFPNet family** | Restoration | `NAFBlock`, `kernel_attention`; variants `UFPNet`, `UFPNet_code_uncertainty`, `Baseline` | `AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py`, `AdaRevD/UFPNet/Baseline_arch.py` |
| 3 | **CAMixer family** | SR | `CAMixer`, `GatedFeedForward`; variants `CAMixerSR`, `CAMixerOSR`, `ClassSR_3class_CAMixerSR_net` | `CAMixer/CAMixerSR_arch.py`, `CAMixer/CAMixerOSR_arch.py`, `CAMixer/ClassSR_CAMixerSR_arch.py` |
| 4 | **CFAT** | SR | `WindowAttention_D/S`, `RHAG` | `CFAT/cfat.py` (`CFAT`) |
| 5 | **ConvMambaSR** | SR | `SS2D`, `VSSBlock`, `PatchEmbed` | `ConvMambaSR/ConvMambaSR_arch.py` (`ConvMambaSR`) |
| 6 | **PromptIR / Restormer (DFPIR)** | Restoration | `PromptGenBlock`, `TransformerBlock`, `Restormer` | `DFPIR/dfpir_arch.py` (`PromptIR`, `Restormer`) |
| 7 | **MSDeformableNAIR** | Restoration | `MSDeformableNeighborhoodAttention`, `TransformerBlock` | `DSwinIR/MSDeformableNAIR.py` (`MSDeformableNAIR`) |
| 8 | **DeblurDiNAT** | Deblurring | `NADeblurMini`, `TransBlock`, baseline variant in `Baseline.py` | `DeblurDiNAT/DeblurDiNAT.py`, `DeblurDiNAT/Baseline.py` |
| 9 | **DiMoSR** | SR | `DiMoSR`, `ResBottleneck` | `DiMoSR/dimosr_arch.py` (`DiMoSR`) |
| 10 | **ESC family** | SR | `ConvolutionalAttention`, `WindowAttention`; variants `ESC`, `ESCFP`, `ESCReal` | `ESC/esc_arch.py`, `ESC/esc_fp_arch.py`, `ESC/esc_real_arch.py` |
| 11 | **RRDB_Net (ESRGANplus)** | SR | `RRDB_Net`, `RRDB` blocks | `ESRGANplus/ESRGANplus_arch.py` (`RRDB_Net`) |
| 12 | **EvTexture** | Restoration | `EvTexture` with UNet/SpyNet alignment modules | `EvTexture/evtexture_arch.py` (`EvTexture`), `EvTexture/unet_arch.py`, `EvTexture/spynet_arch.py` |
| 13 | **FAT** | SR | `FAT`, Swin-style attention + `FourierSparseSelfAttention` | `FAT/FAT.py` (`FAT`) |
| 14 | **FDAT** | SR | `FDAT`, `FastSpatialWindowAttention` | `FDAT/fdat.py` (`FDAT`) |
| 15 | **HAIR** | Restoration | `HAIR`, `Hytrans`, `Hyattn` | `HAIR/HAIR_arch.py` (`HAIR`) |
| 16 | **LSTMConvSR** | SR | `VisionLSTM2`, `LSTMConvSR` | `LSTMConvSR/LSTMConvSR_arch.py` (`LSTMConvSR`) |
| 17 | **LoFormer family** | Restoration / deblurring | `LoFormer`, `FNAFNet`, `DeepDeblur` | `LoFormer/LoFormer_arch.py`, `LoFormer/FNAFNet_arch.py`, `LoFormer/DeepDeblur_arch.py` |
| 18 | **MEASNet (IRmodel)** | Restoration | `Taskprompt`, `TransformerBlock` | `MEASNet/MEASNet.py` (`IRmodel`) |
| 19 | **MP-HSIR** | Hyperspectral restoration | `MP_HSIR_Net`, `CrossTransformer` | `MP-HSIR/MP_HSIR.py` (`MP_HSIR_Net`) |
| 20 | **MoCEIR** | Restoration | `MoCEIR`, `ModExpert`, `RoutingFunction` | `MoCE-IR/moce_ir.py` (`MoCEIR`) |
| 21 | **NEXUSLite** | SR | `EfficientWindowAttention`, `NEXUSLite` | `Nexus/nexus_lite_arch.py` (`NEXUSLite`) |
| 22 | **OAPT_gt** | SR | `SwinTransformerBlock`, `OAPT_gt` | `OAPT/oapt_gt_arch.py` (`OAPT_gt`) |
| 23 | **OSRT** | SR | `WindowAttention`, `OSRT` | `OSRT/osrt_arch.py` (`OSRT`) |
| 24 | **PFT** | SR | `PFTransformerLayer`, `PFT` | `PFT-SR/pft_arch.py` (`PFT`) |
| 25 | **PoolNet / PromptIR (PoolNet)** | Restoration | `PoolNet` encoder/decoder, `PromptIR` transformer | `PoolNet/PoolNet.py` (`PoolNet`), `PoolNet/PoolNet_arch.py` (`PromptIR`) |
| 26 | **RBaIR** | Restoration | `DeformAttn`, `Channel_Cross_Attention` | `RBaIR/RBaIR.py` (`RBaIR`) |
| 27 | **RestorMixer** | Restoration | `Encoder`, `Decoder`, `RestorMixer` | `RestorMixer/model.py` (`RestorMixer`) |
| 28 | **Restore_RWKV** | Restoration | `VRWKV_SpatialMix`, `Restore_RWKV` | `Restore-RWKV/Restore_RWKV.py` (`Restore_RWKV`) |
| 29 | **SFHformer family** | Restoration / deblurring | `Backbone`, `FourierUnit`; motion-blur variant `Backbone_new` | `SFHformer/SFHformer.py`, `SFHformer/sfhformer_motion_blur.py` |
| 30 | **AdaIR** | Restoration | `FreModule`, `Chanel_Cross_Attention`, `AdaIR` | `SIPL/adair_arch.py` (`AdaIR`) |
| 31 | **SeemoRe** | Restoration | `SeemoRe`, MoE (`Expert`, `Router`) | `SeeMore/seemore_arch.py` (`SeemoRe`) |
| 32 | **UVM-Net** | Restoration | `UNet`, `UVMB` | `UVM-Net/model.py` (`UNet`), `UVM-Net/uvmb.py` (`UVMB`) |
| 33 | **VLUNet** | Restoration | `VLUNet`, `CrossAttention` | `VLU-Net/vlu-net_arch.py` (`VLUNet`) |
| 34 | **DetailRefinerNet** | Restoration | `EnhancedRefinementBlock`, `DetailRefinerNet` | `arches/detailrefinernet_arch.py` (`DetailRefinerNet`) |
| 35 | **EMT** | Restoration | `TransformerGroup`, `EMT` | `arches/emt_arch.py` (`EMT`) |
| 36 | **ParagonSR** | SR | `ParagonBlock`, `GatedFFN` | `arches/paragonsr_arch.py` (`ParagonSR`) |
| 37 | **ParagonSR2** | SR | `ATDTransformerLayer`, `AdaptiveTokenCA` | `arches/paragonsr2_arch.py` (`ParagonSR2`) |
| 38 | **ElysiumSR** | SR | `ResidualBlock`, `ElysiumSR` | `arches/elysiumsr_arch.py` (`ElysiumSR_*`) |
| 39 | **DIS** | SR | `DepthwiseSeparableConv`, `PixelShuffleUpsampler` | `arches/dis_arch.py` (`DIS`) |
| 40 | **catanet** | SR | `TAB`, `Attention`, `catanet` | `arches/catanet_arch.py` (`catanet`) |
| 41 | **HyperionSR** | SR | `HyperionBlock`, `GatedFFN` | `arches/hyperionsr_arch.py` (`HyperionSR_*`) |
| 42 | **LKFMixer** | SR | `PLKB`, `PixelShuffleDirect`, `LKFMixer` | `arches/lkfmixer_arch.py` (`LKFMixer`) |
| 43 | **TSPANv2** | Temporal SR / restoration | `TemporalSPANBlock`, `TSPANv2` | `arches/temporal_span_v2_arch.py` (`TSPANv2`) |
| 44 | **ATD** | SR | `ATDTransformerLayer`, `ATD` | `atd/arch.py` (`ATD`) |
| 45 | **DRCT** | SR | `SwinTransformerBlock`, `RDG`, `DRCT` | `drct/DRCT_arch.py` (`DRCT`) |
| 46 | **FlexNet** | Restoration | `TransformerBlock`, `FlexNet` | `flexnet/flexnet_arch.py` (`FlexNet`) |
| 47 | **GateRV3** | SR / restoration | `GatedCNNBlock`, `GateRV3` | `gaterv3/gaterv3_arch.py` (`GateRV3`) |
| 48 | **GFISRV2** | SR | `FourierUnit`, `GFISRV2` | `gfisrv2/gfisrv2_arch.py` (`GFISRV2`) |
| 49 | **GRL** | SR | `TransformerStage`, `GRL` | `grl/grl.py` (`GRL`) |
| 50 | **MoESR** | SR | `MSG`, `MoESR` | `moesr/arch.py` (`MoESR`) |
| 51 | **MoSRv2** | SR | `GatedCNNBlock`, `MoSRv2` | `mosrv2/arch.py` (`MoSRv2`) |
| 52 | **Aether** | SR | `ReparamLargeKernelConv`, `aether` | `neosr/aether_arch.py` (`aether`) |
| 53 | **CFSR** | SR | `LargeKernelConv`, `cfsr` | `neosr/cfsr_arch.py` (`cfsr`) |
| 54 | **EQRSRGAN** | SR (GAN) | `RRDB`, `eqrsrgan` | `neosr/eqrsrgan_arch.py` (`eqrsrgan`) |
| 55 | **EA2FPN** | Restoration / SR | `FPNBlock`, `ea2fpn` | `neosr/ea2fpn_arch.py` (`ea2fpn`) |
| 56 | **HMA** | SR | `WindowAttention`, `hma` | `neosr/hma_arch.py` (`hma`) |
| 57 | **HIT-SRF** | SR | `HierarchicalTransformerBlock`, `hit_srf` | `neosr/hitsrf_arch.py` (`hit_srf`) |
| 58 | **MFGHMoE** | SR | `HMoE`, `mfghmoe` | `neosr/mfghmoe_arch.py` (`mfghmoe`) |
| 59 | **HiFaceGAN** | Face restoration | `HiFaceGAN`, `SPADEGenerator` | `others/hifacegan_arch.py` (`HiFaceGAN`) |
| 60 | **StyleGAN2 family** | Face restoration / GAN | `StyleGAN2Generator`, `StyleGAN2Discriminator` (bilinear variant) | `others/stylegan2_arch.py`, `others/stylegan2_bilinear_arch.py` |
| 61 | **PLKSR family** | SR | `PLKBlock`, `plksr` / `realplksr` | `plksr/plksr_arch.py`, `plksr/realplksr_arch.py` |
| 62 | **RHA** | SR | `HybridAttention`, `RHA` | `rha/arch.py` (`RHA`) |
| 63 | **SpanC** | SR | `SPAB`, `IGConv`, `SpanC` | `spanpp/spanpp_arch.py` (`SpanC`) |
| 64 | **MicroSR** | SR | `SwinTransformerBlock`, `MicroSR` | `team07_MicroSR/MicroSR_Model.py` (`MicroSR`) |
| 65 | **DAT family** | SR | `DAT`, `MSHAT` | `team15_BBox/model.py` (`DAT`), `team15_BBox/MSHAT_model.py` (`MSHAT`) |
| 66 | **HAT family** | SR | `HAT`, `HATM`, `HATIQCMix` | `team18_XiaomiMM/model_2.py`, `team18_XiaomiMM/model_3.py`, `team18_XiaomiMM/model_1.py` |

## Architectures 1–10: meta-information index

### 1) AdaRevID
* **Key features:** Multi-decoder architecture with NAF-style blocks (including FFT-enhanced FCNAFBlock variants), kernel-aware attention, and a UFPNet encoder; includes a classifier/early-exit mechanism across sub-decoders.【F:AdaRevD/AdaRevID_arch.py†L21-L175】【F:AdaRevD/AdaRevID_arch.py†L566-L707】【F:AdaRevD/AdaRevID_arch.py†L879-L936】
* **Operating principle:** Frozen encoder extracts features, then a stack of sub-decoders iteratively refines restoration; early-exit thresholds can skip later decoders for speed.【F:AdaRevD/AdaRevID_arch.py†L566-L707】【F:AdaRevD/AdaRevID_arch.py†L879-L936】
* **Speed/compute:** Potentially heavy due to multiple decoders, but early-exit routing enables variable compute; FFT blocks add extra compute relative to pure-conv baselines.【F:AdaRevD/AdaRevID_arch.py†L566-L707】【F:AdaRevD/AdaRevID_arch.py†L879-L936】
* **Size/memory:** Larger footprint from multiple sub-decoders, classifier heads, and per-stage feature parameters; memory scales with decoder count and intermediate feature maps.【F:AdaRevD/AdaRevID_arch.py†L566-L727】
* **Textures:** Frequency-domain blocks and kernel-aware attention give it strong texture/detail recovery, especially for blur kernels or high-frequency artifacts.【F:AdaRevD/AdaRevID_arch.py†L100-L175】【F:AdaRevD/AdaRevID_arch.py†L566-L707】
* **Advantages:** Flexible compute/quality trade-off via early exit; strong detail modeling with FFT + kernel cues.【F:AdaRevD/AdaRevID_arch.py†L100-L175】【F:AdaRevD/AdaRevID_arch.py†L879-L936】
* **Cons:** Heavy and complex to tune; multiple stages increase training/inference memory and runtime without early-exit savings.【F:AdaRevD/AdaRevID_arch.py†L566-L727】
* **Schematic:** Input → UFPNet encoder → (sub-decoder 1 → … → sub-decoder N with exit classifier) → output head.【F:AdaRevD/AdaRevID_arch.py†L566-L727】

### 2) UFPNet family
* **Key features:** NAFNet-inspired blocks with simplified channel attention, kernel-aware attention (NAFBlock_kernel), and a learned kernel prior module (KernelPrior).【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L37-L186】【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L216-L372】
* **Operating principle:** Encoder–decoder built from NAFBlocks; kernel attention conditions feature processing on blur kernels estimated by KernelPrior.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L98-L186】【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L216-L372】
* **Speed/compute:** Mostly convolutional and depthwise ops, typically faster than transformer-style architectures; kernel attention adds modest overhead.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L37-L186】
* **Size/memory:** Moderate parameter count (conv-heavy) and predictable memory; scales with depth/width more linearly than attention-heavy models.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L216-L372】
* **Textures:** Kernel-aware attention helps recover blur-induced texture loss; NAF blocks favor local texture fidelity.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L98-L186】
* **Advantages:** Stable, efficient restoration baseline; explicit kernel conditioning for deblurring/uncertainty-aware tasks.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L98-L186】【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L216-L372】
* **Cons:** Less global context modeling than transformer or deformable attention models; texture improvements are mostly local.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L37-L186】
* **Schematic:** Input → NAF-style encoder (kernel-aware blocks) → bottleneck → decoder → output.【F:AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py†L216-L372】

### 3) CAMixer family
* **Key features:** Content-adaptive mixing using CAMixer with learned offsets, Gumbel-softmax routing, and a predictor that outputs spatial/channel attention and token selection masks.【F:CAMixer/CAMixerSR_arch.py†L74-L152】【F:CAMixer/CAMixerSR_arch.py†L154-L200】
* **Operating principle:** Features are split into routed/selected tokens; deformable offsets and attention gates modulate mixing within local windows, enabling sparse dynamic computation.【F:CAMixer/CAMixerSR_arch.py†L74-L152】【F:CAMixer/CAMixerSR_arch.py†L154-L200】
* **Speed/compute:** Potentially faster than dense attention by selecting a subset of tokens; deformable routing adds overhead but can reduce overall attention cost.【F:CAMixer/CAMixerSR_arch.py†L74-L200】
* **Size/memory:** Moderate, with extra predictors and routing metadata; memory depends on routing ratio and window size rather than global sequence length.【F:CAMixer/CAMixerSR_arch.py†L74-L200】
* **Textures:** Local window mixing plus learned offsets helps capture textured regions without full global attention, preserving details in SR outputs.【F:CAMixer/CAMixerSR_arch.py†L154-L200】
* **Advantages:** Adaptive compute, better texture focus via token selection, and deformable receptive fields for complex patterns.【F:CAMixer/CAMixerSR_arch.py†L74-L200】
* **Cons:** More complex training/inference pipeline; routing introduces nondeterminism and sensitivity to ratio/hyperparameters.【F:CAMixer/CAMixerSR_arch.py†L74-L200】
* **Schematic:** Input → local window feature extraction → CAMixer routing (offsets + mask) → mixed features → SR head.【F:CAMixer/CAMixerSR_arch.py†L154-L200】

### 4) CFAT
* **Key features:** Window-based attention (shifted and non-shifted), residual hybrid attention groups (RHAG), and channel attention blocks (CAB).【F:CFAT/cfat.py†L67-L155】【F:CFAT/cfat.py†L989-L1070】
* **Operating principle:** Swin-style local attention within windows, grouped into RHAG blocks for hierarchical feature refinement, with CAB for channel recalibration.【F:CFAT/cfat.py†L67-L155】【F:CFAT/cfat.py†L989-L1070】
* **Speed/compute:** Heavier than pure CNNs due to window attention and multiple RHAG stacks; scales with window size rather than full image size.【F:CFAT/cfat.py†L102-L155】【F:CFAT/cfat.py†L989-L1070】
* **Size/memory:** Moderate-to-large depending on RHAG depth; attention caches and relative position bias add overhead vs conv-only designs.【F:CFAT/cfat.py†L102-L155】【F:CFAT/cfat.py†L989-L1070】
* **Textures:** Window attention captures localized textures and repeated patterns; CAB enhances texture by channel-wise reweighting.【F:CFAT/cfat.py†L67-L155】
* **Advantages:** Strong local modeling with transformer-like inductive bias; good balance of texture and structure refinement.【F:CFAT/cfat.py†L102-L155】【F:CFAT/cfat.py†L989-L1070】
* **Cons:** Higher compute/memory than CNN baselines; window partitioning can miss long-range context without deeper stacks.【F:CFAT/cfat.py†L102-L155】【F:CFAT/cfat.py†L989-L1070】
* **Schematic:** Input → window attention blocks (shifted/non-shifted) → RHAG stacks → reconstruction head.【F:CFAT/cfat.py†L102-L155】【F:CFAT/cfat.py†L989-L1070】

### 5) ConvMambaSR
* **Key features:** Combines Mamba selective scan (state-space model) with convolutional and attention modules, including dynamic position bias attention and spatial/channel attention mixers.【F:ConvMambaSR/ConvMambaSR_arch.py†L1-L35】【F:ConvMambaSR/ConvMambaSR_arch.py†L63-L170】
* **Operating principle:** Parallel conv and Mamba streams exchange information via attention modules (e.g., GlobalDetailReconstructionModule), aiming to blend global sequence modeling with local convolutions.【F:ConvMambaSR/ConvMambaSR_arch.py†L115-L170】
* **Speed/compute:** More efficient than full global attention, but still heavier than plain CNNs due to Mamba scan and attention blocks.【F:ConvMambaSR/ConvMambaSR_arch.py†L1-L35】【F:ConvMambaSR/ConvMambaSR_arch.py†L115-L170】
* **Size/memory:** Moderate-to-high; includes additional attention/SSM parameters and multiple attention-style modules beyond standard residual stacks.【F:ConvMambaSR/ConvMambaSR_arch.py†L63-L170】
* **Textures:** Multi-scale spatial attention (MSSA) and channel attention augment texture modeling; Mamba path adds broader context for coherent detail reconstruction.【F:ConvMambaSR/ConvMambaSR_arch.py†L63-L170】
* **Advantages:** Mixes local and global modeling without full attention cost; flexible for SR texture/detail recovery.【F:ConvMambaSR/ConvMambaSR_arch.py†L63-L170】
* **Cons:** Implementation complexity and reliance on Mamba kernels; heavier than compact CNN SR models.【F:ConvMambaSR/ConvMambaSR_arch.py†L1-L35】【F:ConvMambaSR/ConvMambaSR_arch.py†L63-L170】
* **Schematic:** Input → conv trunk + Mamba selective scan path → attention-based fusion → SR reconstruction head.【F:ConvMambaSR/ConvMambaSR_arch.py†L1-L35】【F:ConvMambaSR/ConvMambaSR_arch.py†L115-L170】

### 6) PromptIR / Restormer (DFPIR)
* **Key features:** Restormer-style transformer blocks with MDTA attention and GDFN feed-forward; PromptIR adds prompt generation blocks that inject learned prompts at multiple scales.【F:DFPIR/dfpir_arch.py†L71-L178】【F:DFPIR/dfpir_arch.py†L191-L239】【F:DFPIR/dfpir_arch.py†L493-L540】
* **Operating principle:** Overlap-patch embedding → multi-stage transformer with down/up-sampling; PromptIR uses prompts conditioned on image embeddings to modulate restoration.【F:DFPIR/dfpir_arch.py†L151-L239】【F:DFPIR/dfpir_arch.py†L493-L540】
* **Speed/compute:** Transformer attention is heavier than conv-only architectures; PromptIR adds extra prompt parameters and attention overhead.【F:DFPIR/dfpir_arch.py†L71-L178】【F:DFPIR/dfpir_arch.py†L191-L239】
* **Size/memory:** Larger than NAF-style CNNs due to multi-head attention and prompt tensors; memory scales with spatial resolution and head count.【F:DFPIR/dfpir_arch.py†L71-L178】【F:DFPIR/dfpir_arch.py†L191-L239】
* **Textures:** Attention and prompts help recover fine textures and adapt to diverse degradations; overlap embedding preserves local detail.【F:DFPIR/dfpir_arch.py†L151-L239】
* **Advantages:** Strong general-purpose restoration with prompt conditioning; flexible across multiple degradation types.【F:DFPIR/dfpir_arch.py†L191-L239】【F:DFPIR/dfpir_arch.py†L493-L540】
* **Cons:** Higher compute and memory than lightweight SR networks; training requires careful tuning of prompt layers.【F:DFPIR/dfpir_arch.py†L71-L239】
* **Schematic:** Input → overlap patch embed → transformer stages (+ prompts) → upsampling → output.【F:DFPIR/dfpir_arch.py†L151-L239】【F:DFPIR/dfpir_arch.py†L493-L540】

### 7) MSDeformableNAIR
* **Key features:** Transformer blocks built around MSDeformableNeighborhoodAttention, with Restormer-like attention+FFN structure and overlap patch embedding.【F:DSwinIR/MSDeformableNAIR.py†L71-L152】【F:DSwinIR/MSDeformableNAIR.py†L156-L224】
* **Operating principle:** Local deformable attention aggregates neighborhood features with adaptive sampling, followed by GDFN feed-forward; multi-scale down/up-sampling matches image restoration tasks.【F:DSwinIR/MSDeformableNAIR.py†L116-L178】【F:DSwinIR/MSDeformableNAIR.py†L206-L246】
* **Speed/compute:** More efficient than global attention but costlier than standard window attention due to deformable sampling and extra attention ops.【F:DSwinIR/MSDeformableNAIR.py†L116-L178】
* **Size/memory:** Moderate-to-high; attention maps and deformable kernels add overhead beyond CNN-only methods.【F:DSwinIR/MSDeformableNAIR.py†L71-L178】
* **Textures:** Deformable neighborhoods adapt to edges and textures, improving detail reconstruction on complex blur/noise patterns.【F:DSwinIR/MSDeformableNAIR.py†L116-L178】
* **Advantages:** Adaptive receptive fields and strong local detail modeling; better at non-uniform degradations than fixed-window attention.【F:DSwinIR/MSDeformableNAIR.py†L116-L178】
* **Cons:** Heavier compute and implementation complexity; attention locality may still limit very long-range interactions.【F:DSwinIR/MSDeformableNAIR.py†L116-L178】
* **Schematic:** Input → overlap embed → deformable-attention transformer blocks → up/down sampling → output.【F:DSwinIR/MSDeformableNAIR.py†L156-L246】

### 8) DeblurDiNAT
* **Key features:** Neighborhood Attention (NAT) via NeighborhoodAttention2D, conditional positional embedding (PEG), and lightweight gated/DMFN feed-forward modules in TransBlock.【F:DeblurDiNAT/DeblurDiNAT.py†L38-L90】【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Operating principle:** Local neighborhood attention focuses on deblurring in spatial windows, with PEG adding positional bias and gated feed-forward refinement per block.【F:DeblurDiNAT/DeblurDiNAT.py†L38-L90】【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Speed/compute:** NAT is heavier than pure conv but cheaper than global attention; compute scales with neighborhood kernel size and dilation.【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Size/memory:** Moderate; attention requires temporary buffers per neighborhood but is bounded by local window size.【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Textures:** Neighborhood attention is well-suited to local texture reconstruction after blur, with dilation extending receptive field for fine detail.【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Advantages:** Strong local restoration for deblurring; PEG and channel modulation stabilize alignment of local details.【F:DeblurDiNAT/DeblurDiNAT.py†L68-L90】【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Cons:** Limited global context and heavier than CNN-only deblurring baselines; kernel sizes must be tuned for performance vs cost.【F:DeblurDiNAT/DeblurDiNAT.py†L114-L189】
* **Schematic:** Input → embeddings → NAT-based TransBlocks (+ PEG) → reconstruction head.【F:DeblurDiNAT/DeblurDiNAT.py†L68-L189】

### 9) DiMoSR
* **Key features:** Lightweight CNN SR with multi-stage residual bottlenecks and dilated convolutions, plus stage-wise fusion and PixelShuffle upsampling.【F:DiMoSR/dimosr_arch.py†L8-L45】【F:DiMoSR/dimosr_arch.py†L70-L135】
* **Operating principle:** Three sequential stages of residual bottlenecks refine features; outputs are concatenated and fused before PixelShuffle upscaling.【F:DiMoSR/dimosr_arch.py†L12-L45】
* **Speed/compute:** Fast relative to attention-based SR; mostly standard convs with a few dilated layers and lightweight bottlenecks.【F:DiMoSR/dimosr_arch.py†L12-L45】【F:DiMoSR/dimosr_arch.py†L70-L135】
* **Size/memory:** Small-to-moderate; parameter count depends on num_feat/num_block but remains CNN-scale with limited attention overhead.【F:DiMoSR/dimosr_arch.py†L8-L45】
* **Textures:** Dilated convolutions expand receptive field for textures while keeping compute low; good for local detail but weaker global modeling.【F:DiMoSR/dimosr_arch.py†L70-L135】
* **Advantages:** Simple, efficient, and stable training; predictable memory usage and easy deployment.【F:DiMoSR/dimosr_arch.py†L8-L45】
* **Cons:** Limited long-range context vs attention/SSM models; texture recovery relies on dilation rather than explicit attention.【F:DiMoSR/dimosr_arch.py†L70-L135】
* **Schematic:** Input → shallow conv → stage1/2/3 bottlenecks → concat + fusion → PixelShuffle output.【F:DiMoSR/dimosr_arch.py†L12-L45】

### 10) ESC family
* **Key features:** Combines convolutional attention (dynamic and large-kernel depthwise conv) with window attention supporting multiple backends (naive, SDPA, Flex, FlashBias).【F:ESC/esc_arch.py†L17-L35】【F:ESC/esc_arch.py†L90-L200】
* **Operating principle:** Split channels for convolutional attention (large-kernel + dynamic conv) and windowed self-attention with relative position bias; attention backend can trade speed vs memory.【F:ESC/esc_arch.py†L90-L200】
* **Speed/compute:** Flexible: can run faster with Flex/FlashBias attention or more stable but slower with naive SDPA; conv attention adds overhead vs pure window attention.【F:ESC/esc_arch.py†L17-L35】【F:ESC/esc_arch.py†L90-L200】
* **Size/memory:** Moderate-to-high; dynamic kernels and attention bias tables increase parameters and activation memory, especially at higher resolution.【F:ESC/esc_arch.py†L90-L200】
* **Textures:** Large-kernel conv attention is texture-friendly (captures broader spatial patterns); window attention refines local structures with relative bias.【F:ESC/esc_arch.py†L90-L200】
* **Advantages:** Strong local texture recovery with configurable attention backends; adaptable to different hardware constraints.【F:ESC/esc_arch.py†L17-L35】【F:ESC/esc_arch.py†L90-L200】
* **Cons:** Complexity and backend-specific behavior; training/testing choices affect reproducibility and performance.【F:ESC/esc_arch.py†L17-L35】
* **Schematic:** Input → conv-attention + window-attention blocks → reconstruction head (ESC/ESCFP/ESCReal variants).【F:ESC/esc_arch.py†L90-L200】

## Architectures 11–20: meta-information index

### 11) RRDB_Net (ESRGANplus)
* **Key features:** Stacks of RRDB blocks (Residual-in-Residual Dense Blocks) with residual dense sub-blocks, followed by upsampling (upconv or pixelshuffle) and HR reconstruction convs.【F:ESRGANplus/ESRGANplus_arch.py†L7-L35】【F:ESRGANplus/block.py†L235-L271】【F:ESRGANplus/block.py†L275-L310】
* **Operating principle:** Shallow conv → deep RRDB trunk with residual scaling → LR conv → multi-step upsampling → HR conv output.【F:ESRGANplus/ESRGANplus_arch.py†L12-L35】【F:ESRGANplus/block.py†L235-L271】
* **Speed/compute:** Pure convolutional dense blocks; typically faster than transformer-style SR, but compute grows with RRDB depth (`nb`) and growth channels (`gc`).【F:ESRGANplus/ESRGANplus_arch.py†L7-L35】【F:ESRGANplus/block.py†L200-L271】
* **Size/memory:** Moderate-to-large due to dense connections inside each RRDB; memory scales with feature width and block count but remains predictable vs attention models.【F:ESRGANplus/ESRGANplus_arch.py†L7-L35】【F:ESRGANplus/block.py†L200-L271】
* **Textures:** Dense residual mixing and deep conv stacks emphasize high-frequency detail reconstruction; no explicit global attention, so textures are local but rich.【F:ESRGANplus/block.py†L200-L271】
* **Advantages:** Stable and classic SR baseline; strong local detail recovery with simple, deployable conv-only pipeline.【F:ESRGANplus/ESRGANplus_arch.py†L7-L35】
* **Cons:** Limited global context vs attention/SSM approaches; quality gains scale with depth, increasing runtime and memory.【F:ESRGANplus/ESRGANplus_arch.py†L7-L35】
* **Schematic:** Input → fea_conv → RRDB stack → LR_conv → upsample blocks → HR_conv → output.【F:ESRGANplus/ESRGANplus_arch.py†L12-L35】

### 12) EvTexture
* **Key features:** Dual-branch propagation with RGB flow alignment (SpyNet) and event-voxel texture enhancement via UNet + iterative update block, plus forward/backward trunks and 4× PixelShuffle reconstruction.【F:EvTexture/evtexture_arch.py†L12-L66】【F:EvTexture/evtexture_arch.py†L73-L176】
* **Operating principle:** Estimate optical flow between frames, propagate features backward/forward, and refine texture using event voxel grids through iterative updates; combine motion and texture branches before SR reconstruction.【F:EvTexture/evtexture_arch.py†L73-L176】
* **Speed/compute:** Heavy for VSR due to per-frame flow estimation, iterative voxel updates (per bin), and dual propagation; more expensive than single-image CNN SR.【F:EvTexture/evtexture_arch.py†L33-L66】【F:EvTexture/evtexture_arch.py†L73-L176】
* **Size/memory:** Higher memory from multi-frame features, flow fields, and iterative hidden states; scales with number of frames and voxel bins.【F:EvTexture/evtexture_arch.py†L73-L176】
* **Textures:** Event-driven texture branch explicitly targets fine detail and motion boundaries; strong on fast-motion textures where RGB alone blurs.【F:EvTexture/evtexture_arch.py†L73-L176】
* **Advantages:** Combines motion alignment and event textures for sharper VSR; robust to motion blur when event data is available.【F:EvTexture/evtexture_arch.py†L73-L176】
* **Cons:** Requires event voxel inputs; computationally heavy and more complex training/inference pipeline vs SR-only models.【F:EvTexture/evtexture_arch.py†L73-L176】
* **Schematic:** Frames + event voxels → SpyNet flow + UNet event refinement → backward/forward propagation → PixelShuffle upsampling → output frames.【F:EvTexture/evtexture_arch.py†L33-L176】

### 13) FAT
* **Key features:** Swin-style window attention with relative position bias, channel/spatial attention, and FourierSparseSelfAttention; uses SRT blocks and UAFB for feature refinement.【F:FAT/FAT.py†L1-L120】【F:FAT/FAT.py†L340-L433】
* **Operating principle:** Patch embed → stacked SRT blocks combining local window attention and Fourier attention → UAFB refinement → convolutional fusion and upsampling for SR.【F:FAT/FAT.py†L340-L433】
* **Speed/compute:** Heavier than CNNs due to attention and Fourier modules; windowed attention keeps cost sub-quadratic but still larger than conv-only SR.【F:FAT/FAT.py†L1-L120】【F:FAT/FAT.py†L340-L433】
* **Size/memory:** Moderate-to-high with attention projections, position bias tables, and extra attention modules (CA/SA/FSSA).【F:FAT/FAT.py†L1-L120】
* **Textures:** Fourier attention + spatial/channel attention enhance high-frequency texture modeling and repeated patterns.【F:FAT/FAT.py†L90-L120】【F:FAT/FAT.py†L340-L433】
* **Advantages:** Strong texture/detail recovery with both spatial attention and frequency-domain cues; suitable for high-quality SR at the cost of compute.【F:FAT/FAT.py†L1-L120】【F:FAT/FAT.py†L340-L433】
* **Cons:** Larger runtime and memory than compact CNN SR; window attention still limits very long-range interactions without deeper stacks.【F:FAT/FAT.py†L1-L120】【F:FAT/FAT.py†L340-L433】
* **Schematic:** Input → PatchEmbed → SRT block stack (window attention + FSSA) → UAFB → conv + upsample → output.【F:FAT/FAT.py†L340-L433】

### 14) FDAT
* **Key features:** Simplified dual attention blocks mixing fast spatial window attention and fast channel attention, with depthwise conv and AIM-based fusion inside residual groups; UniUpsampleV3 output head.【F:FDAT/fdat.py†L340-L435】
* **Operating principle:** Shallow conv → multiple residual groups of alternating spatial/channel attention blocks → conv fusion → unified upsampler for SR.【F:FDAT/fdat.py†L360-L455】
* **Speed/compute:** Designed as a faster transformer variant; attention is windowed and simplified, typically lighter than full Swin stacks but heavier than CNNs.【F:FDAT/fdat.py†L340-L455】
* **Size/memory:** Moderate, with attention and AIM modules; scales with groups and block depth rather than full global attention length.【F:FDAT/fdat.py†L360-L455】
* **Textures:** Dual spatial/channel attention improves local texture and edge fidelity compared to pure CNNs.【F:FDAT/fdat.py†L340-L425】
* **Advantages:** Balanced speed/quality with simplified attention; modular residual groups ease scaling for light/tiny variants.【F:FDAT/fdat.py†L360-L455】
* **Cons:** Still more complex than CNN SR; window attention may miss global context on very large structures.【F:FDAT/fdat.py†L360-L455】
* **Schematic:** Input → conv_first → residual groups (spatial/channel attention + AIM) → conv_after → UniUpsampleV3 → output.【F:FDAT/fdat.py†L360-L455】

### 15) HAIR
* **Key features:** Multi-level encoder–decoder with HyLevel blocks, Hytrans/Hyattn-style attention, and feature-conditioned hyper-convolutions; uses a ResNet feature extractor to condition latent/decoder processing.【F:HAIR/HAIR_arch.py†L14-L120】【F:HAIR/HAIR_arch.py†L121-L176】
* **Operating principle:** Overlap patch embedding → hierarchical encoder with downsampling → latent HyLevel conditioned on extracted features → symmetric decoder with skip connections and refinement → residual output.【F:HAIR/HAIR_arch.py†L32-L120】
* **Speed/compute:** Heavy due to multi-scale transformer blocks, hyper-convolutional attention, and deep refinement stages; slower than lightweight CNN SR/restoration models.【F:HAIR/HAIR_arch.py†L32-L120】
* **Size/memory:** Large memory footprint from multi-scale features and hyper-network conditioning; scales with block counts and head sizes.【F:HAIR/HAIR_arch.py†L32-L120】
* **Textures:** Attention-driven blocks and feature-conditioned processing enhance texture and structure detail, especially in restoration tasks with complex degradations.【F:HAIR/HAIR_arch.py†L32-L120】
* **Advantages:** Strong hierarchical modeling with adaptive conditioning; good for complex restoration with multi-scale context.【F:HAIR/HAIR_arch.py†L32-L120】
* **Cons:** High compute and memory; complexity makes deployment and tuning harder than simpler baselines.【F:HAIR/HAIR_arch.py†L32-L120】
* **Schematic:** Input → patch embed → encoder levels (downsample) → HyLevel latent + feature conditioning → decoder levels (upsample + skips) → refinement → output + residual.【F:HAIR/HAIR_arch.py†L32-L120】

### 16) LSTMConvSR
* **Key features:** Hybrid blocks combining Vision LSTM-style attention (ViLBlockPair) with CNN residual stacks, fused by conv; deep residual groups with upsampling heads.【F:LSTMConvSR/LSTMConvSR_arch.py†L43-L120】【F:LSTMConvSR/LSTMConvSR_arch.py†L120-L220】
* **Operating principle:** Shallow conv → repeated blocks that process features through a sequence-modeling (LSTM-like) branch and CNN branch → fusion → residual groups → PixelShuffle upsampling.【F:LSTMConvSR/LSTMConvSR_arch.py†L43-L120】【F:LSTMConvSR/LSTMConvSR_arch.py†L160-L260】
* **Speed/compute:** Heavier than pure CNNs due to LSTM-style sequence mixing; still likely cheaper than full transformers, but slower than DiMoSR-style CNN SR.【F:LSTMConvSR/LSTMConvSR_arch.py†L43-L120】
* **Size/memory:** Moderate-to-high; dual-branch blocks and residual groups increase parameter count and activations vs CNN-only models.【F:LSTMConvSR/LSTMConvSR_arch.py†L84-L220】
* **Textures:** CNN branch preserves local texture; LSTM-style mixing adds broader context for coherent detail across regions.【F:LSTMConvSR/LSTMConvSR_arch.py†L43-L120】
* **Advantages:** Blends local detail modeling with longer-range dependencies without full attention cost; flexible depth/width scaling.【F:LSTMConvSR/LSTMConvSR_arch.py†L84-L220】
* **Cons:** More complex than standard CNN SR; sequence mixing still adds overhead and may be sensitive to resolution.【F:LSTMConvSR/LSTMConvSR_arch.py†L43-L120】
* **Schematic:** Input → conv_first → (LSTM-like branch + CNN branch) fusion blocks → residual groups → upsampler → output.【F:LSTMConvSR/LSTMConvSR_arch.py†L84-L220】

### 17) LoFormer family
* **Key features:** LoFormer uses window/grid-based transformer blocks with down/up-sampling, while FNAFNet is a NAFNet-style encoder–decoder with FFT-aware blocks; DeepDeblur is a multi-stage pyramid restoration network.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Operating principle:** LoFormer applies overlap patch embedding and stacked transformer blocks across encoder/decoder levels; FNAFNet uses NAF-style blocks (with FFT modules) in a U-Net-like structure; DeepDeblur runs coarse-to-fine stages over an image pyramid with successive refinements.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Speed/compute:** LoFormer and FNAFNet are heavier than pure CNNs due to transformer/FFT blocks; DeepDeblur’s multi-stage pyramid adds extra passes but is conv-only per stage.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Size/memory:** LoFormer/FNAFNet use multi-level features and attention/FFT modules, increasing memory; DeepDeblur scales with pyramid depth and per-stage feature width.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Textures:** FFT-aware FNAFNet blocks and LoFormer attention favor texture/detail recovery; DeepDeblur improves coarse-to-fine texture stabilization across scales.【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Advantages:** Diverse options: LoFormer for attention-heavy restoration, FNAFNet for FFT-enhanced NAF-style efficiency, DeepDeblur for robust multi-scale refinement.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Cons:** Complexity and compute vary by variant; LoFormer/FNAFNet require careful tuning for stability, DeepDeblur adds latency due to multi-stage pyramid.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】
* **Schematic:** LoFormer: Input → overlap embed → encoder blocks ↓ → middle blocks → decoder blocks ↑ → output. FNAFNet: Input → NAF blocks ↓ → middle → NAF blocks ↑ → output + residual. DeepDeblur: image pyramid → stage1 → upsample → stage2 → upsample → stage3.【F:LoFormer/LoFormer_arch.py†L600-L740】【F:LoFormer/FNAFNet_arch.py†L311-L410】【F:LoFormer/DeepDeblur_arch.py†L140-L200】

### 18) MEASNet (IRmodel)
* **Key features:** Task prompt generation, task-harmonization blocks (STPG_G_MESE), frequency-aware MEE experts (FD_MEE), and a multi-level transformer encoder–decoder backbone.【F:MEASNet/MEASNet.py†L260-L350】
* **Operating principle:** Generate task prompts from input, inject them into encoder stages, process with multi-scale transformer blocks, and refine with expert modules before residual output.【F:MEASNet/MEASNet.py†L260-L350】
* **Speed/compute:** Heavier than plain Restormer/NAFNet due to prompt conditioning and multiple expert modules; compute scales with stage depth and expert count.【F:MEASNet/MEASNet.py†L260-L350】
* **Size/memory:** Larger memory footprint from prompt tensors and expert modules; multi-scale encoder/decoder adds activation overhead.【F:MEASNet/MEASNet.py†L260-L350】
* **Textures:** Frequency-aware modules and task prompts help adapt to different degradations, improving texture recovery across tasks.【F:MEASNet/MEASNet.py†L240-L350】
* **Advantages:** Task-adaptive restoration with explicit prompts and expert routing; flexible across multiple degradation types.【F:MEASNet/MEASNet.py†L260-L350】
* **Cons:** More complex training/inference; extra modules increase latency and memory vs simpler transformer restorers.【F:MEASNet/MEASNet.py†L260-L350】
* **Schematic:** Input → task prompt → encoder (with task harmonization) → latent → decoder (+ expert modules) → refinement → output + residual.【F:MEASNet/MEASNet.py†L260-L350】

### 19) MP-HSIR
* **Key features:** Hyperspectral restoration with prompt-driven attention blocks (PGSSTB), text prompts, and prompt fusion modules at multiple scales.【F:MP-HSIR/MP_HSIR.py†L720-L835】
* **Operating principle:** Overlap patch embedding → multi-level encoder blocks → latent → decoder with prompt fusion conditioned on text prompts → refinement → residual output.【F:MP-HSIR/MP_HSIR.py†L740-L835】
* **Speed/compute:** Attention-heavy with prompt conditioning; heavier than standard CNN hyperspectral models but localized via windowed blocks.【F:MP-HSIR/MP_HSIR.py†L720-L835】
* **Size/memory:** Moderate-to-high; prompt tensors and multi-scale attention blocks increase parameters and activations, scaling with spectral channels and window size.【F:MP-HSIR/MP_HSIR.py†L720-L835】
* **Textures:** Windowed attention and prompt fusion can preserve fine spectral-spatial textures across bands, aiding detailed hyperspectral reconstruction.【F:MP-HSIR/MP_HSIR.py†L740-L835】
* **Advantages:** Task-aware prompts support multi-task hyperspectral restoration; explicit prompt fusion improves adaptation across degradations.【F:MP-HSIR/MP_HSIR.py†L740-L835】
* **Cons:** Requires prompt generation and task IDs; heavier than compact hyperspectral CNNs, with more tuning complexity.【F:MP-HSIR/MP_HSIR.py†L720-L835】
* **Schematic:** Input → patch embed → encoder blocks → latent → decoder + prompt fusion → refinement → output + residual.【F:MP-HSIR/MP_HSIR.py†L740-L835】

### 20) MoCEIR
* **Key features:** Encoder–decoder with frequency embedding, decoder residual groups using mixture-of-experts attention layers, and complexity-aware routing (rank/top-k).【F:MoCE-IR/moce_ir.py†L720-L845】
* **Operating principle:** Encode multi-level features, compute a frequency embedding via high-pass filtering, and decode with expert layers conditioned on the frequency embedding; residual output with refinement.【F:MoCE-IR/moce_ir.py†L720-L845】
* **Speed/compute:** Potentially heavy due to MoE experts and multi-level encoder/decoder; compute can be modulated via rank/top-k routing settings.【F:MoCE-IR/moce_ir.py†L720-L845】
* **Size/memory:** Larger than single-expert models; expert parameters and frequency embedding add overhead, scaling with number of experts and stage depth.【F:MoCE-IR/moce_ir.py†L720-L845】
* **Textures:** Frequency embedding guides expert selection toward high-frequency detail recovery; good for texture-rich restoration tasks.【F:MoCE-IR/moce_ir.py†L720-L845】
* **Advantages:** Adaptive expert routing for diverse degradations; frequency-aware conditioning can improve texture fidelity without full global attention.【F:MoCE-IR/moce_ir.py†L720-L845】
* **Cons:** MoE complexity increases training instability and inference overhead; routing adds extra configuration and memory needs.【F:MoCE-IR/moce_ir.py†L720-L845】
* **Schematic:** Input → patch embed → encoder levels ↓ → latent → frequency embedding → decoder with MoE blocks ↑ → refinement → output + residual.【F:MoCE-IR/moce_ir.py†L720-L845】
