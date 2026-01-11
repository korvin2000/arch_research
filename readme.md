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

## Architectures 21–30: meta-information index

### 21) NEXUSLite
* **Key features:** Memory-optimized transformer with efficient window attention (ALiBi bias), lightweight channel attention, and reduced-expansion FFN; stages alternate spatial and channel blocks for lower attention memory.【F:Nexus/nexus_lite_arch.py†L1-L24】【F:Nexus/nexus_lite_arch.py†L92-L178】【F:Nexus/nexus_lite_arch.py†L360-L446】【F:Nexus/nexus_lite_arch.py†L465-L551】
* **Operating principle:** Shallow conv → patch embedding → stacked stages of alternating spatial/channel blocks → norm + unembed → residual conv → pixel-shuffle upsampler.【F:Nexus/nexus_lite_arch.py†L620-L785】
* **Speed/compute:** Designed for reduced memory/compute via smaller window size and 2× FFN expansion; alternating attention cuts parallel attention cost relative to dual-path designs.【F:Nexus/nexus_lite_arch.py†L1-L24】【F:Nexus/nexus_lite_arch.py†L360-L446】【F:Nexus/nexus_lite_arch.py†L465-L551】
* **Size/memory:** Explicitly targets ~80% memory reduction; compact variants (tiny/small/medium/large) scale depth/width without changing the lightweight block structure.【F:Nexus/nexus_lite_arch.py†L1-L24】【F:Nexus/nexus_lite_arch.py†L780-L896】
* **Textures:** Window attention plus channel attention preserve local textures while keeping compute bounded; ALiBi supports stable attention with lower overhead.【F:Nexus/nexus_lite_arch.py†L92-L178】【F:Nexus/nexus_lite_arch.py†L360-L446】
* **Advantages:** Strong memory efficiency and scalable capacity while retaining transformer-style local modeling.【F:Nexus/nexus_lite_arch.py†L1-L24】【F:Nexus/nexus_lite_arch.py†L465-L551】
* **Cons:** Windowed attention limits long-range global context unless deeper stages are used; still heavier than pure CNN SR for very low-resource targets.【F:Nexus/nexus_lite_arch.py†L92-L178】【F:Nexus/nexus_lite_arch.py†L620-L785】
* **Schematic:** Input → conv_first → patch embed → alternating spatial/channel stages → conv_after_body + residual → PixelShuffle upsampler → output.【F:Nexus/nexus_lite_arch.py†L620-L785】

### 22) OAPT_gt
* **Key features:** Swin-style residual transformer blocks (RSTB) with window attention and optional frequency-enhanced SFB residual connection using FFT-based branches.【F:OAPT/oapt_gt_arch.py†L19-L80】【F:OAPT/oapt_gt_arch.py†L171-L260】【F:OAPT/oapt_gt_arch.py†L910-L1016】
* **Operating principle:** Shallow conv → patch embed → RSTB stacks with window attention → residual conv/SFB → upsampling head (pixelshuffle/nearest+conv) → output.【F:OAPT/oapt_gt_arch.py†L910-L1038】
* **Speed/compute:** Transformer window attention scales with window size; optional SFB adds FFT-based compute but remains local to feature maps.【F:OAPT/oapt_gt_arch.py†L19-L80】【F:OAPT/oapt_gt_arch.py†L171-L260】【F:OAPT/oapt_gt_arch.py†L910-L1016】
* **Size/memory:** Moderate-to-high depending on depth and heads; optional SFB increases parameters and activation memory versus 1conv residual connections.【F:OAPT/oapt_gt_arch.py†L34-L80】【F:OAPT/oapt_gt_arch.py†L969-L1016】
* **Textures:** FFT branch in SFB injects frequency-domain refinement, improving high-frequency texture reconstruction relative to pure window attention.【F:OAPT/oapt_gt_arch.py†L19-L80】【F:OAPT/oapt_gt_arch.py†L998-L1016】
* **Advantages:** Strong local modeling with Swin-style attention and optional frequency-aware refinement for sharper SR outputs.【F:OAPT/oapt_gt_arch.py†L19-L80】【F:OAPT/oapt_gt_arch.py†L910-L1016】
* **Cons:** Attention windows can miss global dependencies; FFT residuals add complexity and extra latency when enabled.【F:OAPT/oapt_gt_arch.py†L19-L80】【F:OAPT/oapt_gt_arch.py†L969-L1016】
* **Schematic:** Input → conv_first → patch embed → RSTB window attention stack → (1conv/3conv/SFB) residual → upsampler → output.【F:OAPT/oapt_gt_arch.py†L910-L1038】

### 23) OSRT
* **Key features:** Swin-style RSTB stages with window attention, optional ViT conditioning, and deformable convolution conditioning via DCNv2-based offsets for restoration-aware modulation.【F:OSRT/osrt_arch.py†L71-L178】【F:OSRT/osrt_arch.py†L802-L980】【F:OSRT/osrt_arch.py†L980-L1080】
* **Operating principle:** Shallow conv → patch embed → RSTB blocks (with optional vit/dcn conditioning) → residual conv → reconstruction head (pixelshuffle/nearest+conv).【F:OSRT/osrt_arch.py†L802-L1080】
* **Speed/compute:** Window attention keeps compute local; enabling DCN conditioning adds offset convs and deformable ops, increasing latency vs vanilla Swin SR.【F:OSRT/osrt_arch.py†L71-L178】【F:OSRT/osrt_arch.py†L980-L1080】
* **Size/memory:** Base model is comparable to Swin SR; conditional branches (vit/dcn) add parameters and activation memory for condition features and offsets.【F:OSRT/osrt_arch.py†L802-L980】【F:OSRT/osrt_arch.py†L980-L1080】
* **Textures:** Window attention preserves local textures, while deformable conditioning can better align structure in challenging degradations (e.g., warps/blur).【F:OSRT/osrt_arch.py†L71-L178】【F:OSRT/osrt_arch.py†L980-L1080】
* **Advantages:** Flexible conditioning for restoration-aware adaptation without abandoning Swin-like efficiency.【F:OSRT/osrt_arch.py†L802-L1080】
* **Cons:** Conditional branches increase complexity and tuning burden; deformable ops add extra memory/compute overhead.【F:OSRT/osrt_arch.py†L980-L1080】
* **Schematic:** Input → conv_first → patch embed → conditioned RSTB stack → conv_after_body → upsampler → output.【F:OSRT/osrt_arch.py†L802-L1080】

### 24) PFT
* **Key features:** Progressive Focused Transformer using top-k sparse window attention, custom CUDA SMM kernels for sparse QK/V ops, and ConvFFN with depthwise spatial mixing.【F:PFT-SR/pft_arch.py†L1-L120】【F:PFT-SR/pft_arch.py†L180-L270】【F:PFT-SR/pft_arch.py†L360-L520】【F:PFT-SR/pft_arch.py†L860-L980】
* **Operating principle:** Shallow conv → patch embed → PFTB stages (PFTransformerLayer with sparse window attention + ConvFFN) → residual conv → upsampling head → output.【F:PFT-SR/pft_arch.py†L360-L740】【F:PFT-SR/pft_arch.py†L860-L1060】
* **Speed/compute:** Sparse top-k attention reduces compute vs dense windows but relies on custom CUDA kernels; cost scales with chosen top-k schedule.【F:PFT-SR/pft_arch.py†L1-L120】【F:PFT-SR/pft_arch.py†L860-L920】
* **Size/memory:** Moderate-to-high; attention parameters plus extra sparse indices/values add overhead relative to CNN SR, but sparsity can reduce activation load.【F:PFT-SR/pft_arch.py†L360-L520】【F:PFT-SR/pft_arch.py†L860-L980】
* **Textures:** Window attention with progressive focusing keeps detail-rich tokens while ConvFFN preserves local texture via depthwise mixing.【F:PFT-SR/pft_arch.py†L360-L520】
* **Advantages:** Adaptive sparsity provides a quality/compute trade-off and focuses capacity on high-detail regions.【F:PFT-SR/pft_arch.py†L360-L520】【F:PFT-SR/pft_arch.py†L860-L980】
* **Cons:** Requires custom CUDA extensions (smm_cuda); tuning top-k schedules is nontrivial and impacts stability/performance.【F:PFT-SR/pft_arch.py†L1-L120】【F:PFT-SR/pft_arch.py†L860-L920】
* **Schematic:** Input → conv_first → patch embed → PFTB stages (sparse window attention + ConvFFN) → conv_after_body → upsampler → output.【F:PFT-SR/pft_arch.py†L360-L740】【F:PFT-SR/pft_arch.py†L860-L1060】

### 25) PoolNet / PromptIR (PoolNet)
* **Key features:** PoolNet uses multi-scale encoder/decoder with residual blocks, SCM/FAM fusion modules, and multi-output supervision; PromptIR variant is a Restormer-style encoder–decoder with MDTA attention, GDFN feed-forward, and optional frequency branches (Avg/Max pooling mixers).【F:PoolNet/PoolNet.py†L6-L126】【F:PoolNet/PoolNet.py†L128-L210】【F:PoolNet/PoolNet_arch.py†L66-L170】【F:PoolNet/PoolNet_arch.py†L320-L520】
* **Operating principle:** PoolNet builds a 3-level pyramid with SCM-conditioned inputs and FAM feature fusion, then decodes with skip connections and multi-scale outputs; PromptIR builds overlap patch embed → transformer encoder/decoder → refinement → residual output.【F:PoolNet/PoolNet.py†L63-L210】【F:PoolNet/PoolNet_arch.py†L343-L520】
* **Speed/compute:** PoolNet is conv-heavy and efficient but uses multi-scale paths; PromptIR is heavier due to multi-head attention and deeper transformer stacks.【F:PoolNet/PoolNet.py†L63-L210】【F:PoolNet/PoolNet_arch.py†L66-L170】【F:PoolNet/PoolNet_arch.py†L343-L520】
* **Size/memory:** PoolNet scales with residual block count (small/base/large); PromptIR scales with transformer depth/heads and decoder refinement blocks.【F:PoolNet/PoolNet.py†L63-L126】【F:PoolNet/PoolNet_arch.py†L343-L520】
* **Textures:** PoolNet’s multi-scale fusion stabilizes textures across scales; PromptIR’s attention + frequency branches emphasize high-frequency detail and structural consistency.【F:PoolNet/PoolNet.py†L128-L210】【F:PoolNet/PoolNet_arch.py†L320-L520】
* **Advantages:** PoolNet offers lightweight multi-scale restoration; PromptIR offers stronger detail modeling via transformer attention and frequency-aware branches.【F:PoolNet/PoolNet.py†L63-L210】【F:PoolNet/PoolNet_arch.py†L343-L520】
* **Cons:** PoolNet lacks global attention; PromptIR is heavier and more memory-intensive than CNN baselines.【F:PoolNet/PoolNet.py†L63-L210】【F:PoolNet/PoolNet_arch.py†L343-L520】
* **Schematic:** PoolNet: Input → multi-scale SCM + encoder → FAM fusion → decoder + multi-scale outputs. PromptIR: Input → overlap patch embed → transformer encoder/decoder → refinement → output + residual.【F:PoolNet/PoolNet.py†L63-L210】【F:PoolNet/PoolNet_arch.py†L343-L520】

### 26) RBaIR
* **Key features:** Restormer-style transformer blocks (MDTA + GDFN), channel cross-attention, deformable attention, and dynamic RBF-based frequency modulation modules (DyRBF) injected in the decoder path.【F:RBaIR/RBaIR.py†L14-L176】【F:RBaIR/RBaIR.py†L188-L300】【F:RBaIR/RBaIR.py†L379-L506】
* **Operating principle:** Overlap patch embed → transformer encoder → latent → decoder with deformable/cross-attention and DyRBF refinement → final refinement blocks → residual output.【F:RBaIR/RBaIR.py†L379-L520】
* **Speed/compute:** Heavier than conv-only restorers due to attention, deformable sampling, and dynamic RBF modulation; compute scales with block counts and DyRBF settings.【F:RBaIR/RBaIR.py†L188-L300】【F:RBaIR/RBaIR.py†L379-L506】
* **Size/memory:** High, with multiple attention modules plus DyRBF parameters; decoder adds extra branches for adaptive refinement.【F:RBaIR/RBaIR.py†L188-L300】【F:RBaIR/RBaIR.py†L379-L506】
* **Textures:** Deformable and cross-attention improve alignment, while DyRBF modules adapt frequency-aware refinement for detail restoration.【F:RBaIR/RBaIR.py†L188-L300】【F:RBaIR/RBaIR.py†L379-L506】
* **Advantages:** Strong adaptive restoration with multiple attention mechanisms and dynamic frequency modulation for challenging degradations.【F:RBaIR/RBaIR.py†L188-L300】【F:RBaIR/RBaIR.py†L379-L506】
* **Cons:** Complexity and compute cost are high; sensitive to hyperparameter choices in DyRBF/deformable settings.【F:RBaIR/RBaIR.py†L188-L300】【F:RBaIR/RBaIR.py†L379-L506】
* **Schematic:** Input → overlap patch embed → transformer encoder ↓ → latent → decoder + DyRBF refinement → refinement blocks → output + residual.【F:RBaIR/RBaIR.py†L379-L520】

### 27) RestorMixer
* **Key features:** Encoder–decoder with CNN stem + residual depth blocks, VSSBlock + SimpleSwinBlock mixers at deeper stages, and SCM/FAM feature fusion; optional deep supervision outputs.【F:RestorMixer/model.py†L6-L120】【F:RestorMixer/model.py†L122-L220】
* **Operating principle:** CNN stem → encoder with residual blocks + mix stages → decoder with mixed stages and skip connections → final conv output (optionally multi-scale outputs).【F:RestorMixer/model.py†L18-L220】
* **Speed/compute:** Hybrid CNN + attention-like mixers; heavier than pure CNNs but lighter than full transformer stacks due to staged mixing and limited window sizes.【F:RestorMixer/model.py†L18-L220】
* **Size/memory:** Moderate; memory scales with base channel and number of mixer blocks, plus optional deep supervision heads.【F:RestorMixer/model.py†L10-L220】
* **Textures:** Mixer stages (VSSBlock + SimpleSwinBlock) and multi-scale fusion help preserve texture while CNN stem stabilizes local edges.【F:RestorMixer/model.py†L18-L220】
* **Advantages:** Balanced hybrid design with multi-scale fusion and mixed token/conv processing for restoration quality.【F:RestorMixer/model.py†L18-L220】
* **Cons:** More complex than standard U-Nets; additional mixer blocks add overhead vs purely convolutional baselines.【F:RestorMixer/model.py†L18-L220】
* **Schematic:** Input → CNN stem → encoder + mix stages → decoder + skip fusion → output (+ deep supervision outputs).【F:RestorMixer/model.py†L18-L220】

### 28) Restore_RWKV
* **Key features:** RWKV-inspired spatial mixing with custom CUDA WKV kernel, OmniShift multi-kernel depthwise reparameterization, and RWKV-style channel mix blocks; U-Net-like encoder/decoder with skip connections.【F:Restore-RWKV/Restore_RWKV.py†L1-L120】【F:Restore-RWKV/Restore_RWKV.py†L121-L220】【F:Restore-RWKV/Restore_RWKV.py†L275-L370】
* **Operating principle:** Patch conv → stacked RWKV Blocks (spatial mix + channel mix) across encoder/decoder levels → refinement → residual output.【F:Restore-RWKV/Restore_RWKV.py†L230-L370】
* **Speed/compute:** Efficient sequential mixing vs full attention, but depends on custom CUDA WKV kernel and OmniShift depthwise conv reparameterization; compute scales with recurrence and block depth.【F:Restore-RWKV/Restore_RWKV.py†L1-L220】
* **Size/memory:** Moderate; RWKV blocks use linear projections and depthwise convolutions, but U-Net depth and refinement blocks add activation memory.【F:Restore-RWKV/Restore_RWKV.py†L121-L220】【F:Restore-RWKV/Restore_RWKV.py†L275-L370】
* **Textures:** OmniShift combines multi-kernel depthwise convs for richer local texture modeling; RWKV mixing adds broader context without dense attention maps.【F:Restore-RWKV/Restore_RWKV.py†L46-L120】【F:Restore-RWKV/Restore_RWKV.py†L121-L220】
* **Advantages:** Sequence-style mixing with efficient kernels; good balance of local texture and context with modest parameter counts.【F:Restore-RWKV/Restore_RWKV.py†L1-L220】
* **Cons:** Requires custom CUDA extension (bi_wkv); kernel availability and compilation can be a deployment hurdle.【F:Restore-RWKV/Restore_RWKV.py†L1-L20】
* **Schematic:** Input → patch conv → RWKV encoder ↓ → latent → RWKV decoder ↑ → refinement → output + residual.【F:Restore-RWKV/Restore_RWKV.py†L275-L370】

### 29) SFHformer family
* **Key features:** Mixer blocks combining local dilated depthwise convs and global Fourier units, channel attention fusion, and multi-stage backbone with down/up sampling and skip connections.【F:SFHformer/SFHformer.py†L62-L220】【F:SFHformer/SFHformer.py†L240-L340】
* **Operating principle:** Patch embedding → Stage blocks (Mixer + FFN) across encoder/decoder stages → upsample with skip fusion → final patch unembed + residual output.【F:SFHformer/SFHformer.py†L240-L340】
* **Speed/compute:** Largely convolutional with FFT-based global mixing; typically faster than full transformers but heavier than pure CNNs due to Fourier units.【F:SFHformer/SFHformer.py†L120-L220】【F:SFHformer/SFHformer.py†L240-L340】
* **Size/memory:** Moderate; memory scales with stage depth/width and FFT activations in FourierUnit blocks.【F:SFHformer/SFHformer.py†L120-L220】【F:SFHformer/SFHformer.py†L300-L340】
* **Textures:** Combines local dilated convs with frequency-domain global mixing for strong texture/detail reconstruction across scales.【F:SFHformer/SFHformer.py†L90-L220】
* **Advantages:** Efficient hybrid of spatial and frequency modeling; strong texture restoration without heavy attention maps.【F:SFHformer/SFHformer.py†L90-L220】
* **Cons:** FFT-based modules add overhead; global context is frequency-biased rather than explicit token attention.【F:SFHformer/SFHformer.py†L120-L220】
* **Schematic:** Input → patch embed → Stage blocks (local mixer + Fourier mixer + FFN) → down/upsample with skips → patch unembed → output + residual.【F:SFHformer/SFHformer.py†L240-L340】

### 30) AdaIR
* **Key features:** Restormer-style transformer encoder–decoder with frequency mining via FreModule (FFT-based high/low separation), channel cross-attention, and adaptive frequency refinement in decoder stages.【F:SIPL/adair_arch.py†L70-L220】【F:SIPL/adair_arch.py†L315-L396】【F:SIPL/adair_arch.py†L410-L520】
* **Operating principle:** Overlap patch embed → transformer encoder → latent → FreModule-conditioned decoder refinement → final refinement blocks → residual output.【F:SIPL/adair_arch.py†L410-L520】
* **Speed/compute:** Heavier than CNNs due to multi-head attention and FFT-based FreModule; compute scales with stage depth and decoder frequency modules.【F:SIPL/adair_arch.py†L70-L220】【F:SIPL/adair_arch.py†L315-L396】【F:SIPL/adair_arch.py†L410-L520】
* **Size/memory:** Moderate-to-high; attention blocks plus frequency mining add parameters and intermediate activations (FFT, masks, cross-attn).【F:SIPL/adair_arch.py†L70-L220】【F:SIPL/adair_arch.py†L315-L396】
* **Textures:** Explicit high/low-frequency decomposition and cross-attention fusion enhance texture/detail recovery across degradations.【F:SIPL/adair_arch.py†L315-L396】
* **Advantages:** Frequency-aware refinement with transformer backbone provides strong restoration quality for diverse degradations.【F:SIPL/adair_arch.py†L315-L520】
* **Cons:** More complex and heavier than NAF-style CNNs; FFT/cross-attn modules increase runtime and tuning burden.【F:SIPL/adair_arch.py†L70-L220】【F:SIPL/adair_arch.py†L315-L396】
* **Schematic:** Input → overlap patch embed → transformer encoder ↓ → latent → FreModule-guided decoder ↑ → refinement → output + residual.【F:SIPL/adair_arch.py†L410-L520】

## Architectures 31–40: meta-information index

### 31) SeemoRe
* **Key features:** Residual groups combine a local MoE block (RME → MoEBlock/Router/Expert with top‑k routing) and a global block (SME → StripedConvFormer using striped depthwise convs), with GatedFFN for feed-forward mixing.【F:SeeMore/seemore_arch.py†L31-L118】【F:SeeMore/seemore_arch.py†L146-L209】【F:SeeMore/seemore_arch.py†L221-L313】
* **Operating principle:** Shallow conv → stacked ResGroups (local MoE + global striped conv) → LayerNorm → conv + residual → PixelShuffle upsampling.【F:SeeMore/seemore_arch.py†L31-L78】【F:SeeMore/seemore_arch.py†L89-L143】
* **Speed/compute:** MoE top‑k routing limits active experts per token, while striped convolutions keep global mixing convolutional rather than full attention; compute scales with expert count and top‑k.【F:SeeMore/seemore_arch.py†L221-L269】
* **Size/memory:** Increases with number of experts and layers; MoELayer holds multiple Expert modules plus Router weights.【F:SeeMore/seemore_arch.py†L244-L269】【F:SeeMore/seemore_arch.py†L287-L313】
* **Textures:** Local MoE calibration + striped global convs emphasize texture recovery with mixed receptive fields and gated FFN refinement.【F:SeeMore/seemore_arch.py†L102-L118】【F:SeeMore/seemore_arch.py†L146-L209】
* **Advantages:** Adaptive expert routing and global striped mixing offer a flexible quality/compute balance for SR textures.【F:SeeMore/seemore_arch.py†L146-L209】【F:SeeMore/seemore_arch.py†L221-L269】
* **Cons:** MoE routing adds complexity and extra parameters; performance depends on top‑k and expert configuration.【F:SeeMore/seemore_arch.py†L221-L269】
* **Schematic:** Input → conv → [ResGroup: MoE local → StripedConvFormer global]×N → norm → conv + residual → PixelShuffle output.【F:SeeMore/seemore_arch.py†L31-L143】

### 32) UVM-Net
* **Key features:** U‑Net backbone with DoubleConv blocks augmented by UVMB (Mamba-based SSM with multiple Mamba modules and softmax mixing) plus residual output.【F:UVM-Net/model.py†L6-L36】【F:UVM-Net/unet_part.py†L7-L37】【F:UVM-Net/uvmb.py†L7-L52】
* **Operating principle:** U‑Net encoder/decoder with skip connections; each DoubleConv pre-processes features via UVMB at 64×64, then reconstructs and adds the input residual.【F:UVM-Net/model.py†L6-L35】【F:UVM-Net/unet_part.py†L7-L55】
* **Speed/compute:** U‑Net conv cost plus Mamba SSM passes and fixed 64×64 interpolation inside DoubleConv; runtime grows with Mamba depth and input size due to interpolation overhead.【F:UVM-Net/unet_part.py†L21-L37】【F:UVM-Net/uvmb.py†L7-L52】
* **Size/memory:** Moderate-to-high for a U‑Net because UVMB instantiates three Mamba modules (including one with d_model=w*h), adding parameter and activation cost.【F:UVM-Net/uvmb.py†L7-L39】
* **Textures:** Skip connections preserve local textures; UVMB sequence mixing adds longer-range context to refine details.【F:UVM-Net/model.py†L16-L35】【F:UVM-Net/uvmb.py†L33-L52】
* **Advantages:** Combines U‑Net stability with SSM mixing for stronger context without full attention.【F:UVM-Net/model.py†L6-L35】【F:UVM-Net/uvmb.py†L7-L52】
* **Cons:** UVMB interpolation and Mamba modules add latency and memory vs pure U‑Net CNN baselines.【F:UVM-Net/unet_part.py†L21-L37】【F:UVM-Net/uvmb.py†L7-L52】
* **Schematic:** Input → U‑Net down path (DoubleConv+UVMB) → up path with skips → out conv + input residual.【F:UVM-Net/model.py†L6-L35】【F:UVM-Net/unet_part.py†L7-L55】

### 33) VLUNet
* **Key features:** Restormer-style attention (MDTA) + GDFN feed-forward blocks, plus degradation-aware cross-attention via GDM with learned prompts and degradation vectors.【F:VLU-Net/vlu-net_arch.py†L57-L140】【F:VLU-Net/vlu-net_arch.py†L167-L260】
* **Operating principle:** Overlap patch embedding → multi-level encoder/decoder of DUN_BaseBlocks (GDM + transformer blocks) with down/upsampling → refinement → output + residual.【F:VLU-Net/vlu-net_arch.py†L262-L371】
* **Speed/compute:** Attention-heavy with multi-level transformer blocks and cross-attention prompts; compute scales with depth and head counts across encoder/decoder stages.【F:VLU-Net/vlu-net_arch.py†L84-L160】【F:VLU-Net/vlu-net_arch.py†L262-L371】
* **Size/memory:** High for large variants due to multiple stages and prompt parameters inside GDM, plus attention activations per scale.【F:VLU-Net/vlu-net_arch.py†L167-L246】【F:VLU-Net/vlu-net_arch.py†L262-L371】
* **Textures:** Attention and degradation-conditioned prompts help refine fine textures under varying degradations while maintaining structural consistency.【F:VLU-Net/vlu-net_arch.py†L167-L246】
* **Advantages:** Degradation-aware conditioning plus transformer backbone yields flexible restoration across noise/blur regimes.【F:VLU-Net/vlu-net_arch.py†L167-L246】【F:VLU-Net/vlu-net_arch.py†L262-L371】
* **Cons:** Heavier than CNN baselines; prompt conditioning and multi-scale attention increase tuning complexity and memory use.【F:VLU-Net/vlu-net_arch.py†L167-L246】【F:VLU-Net/vlu-net_arch.py†L262-L371】
* **Schematic:** Input → patch embed → encoder levels ↓ → latent → decoder levels ↑ → refinement → output + residual.【F:VLU-Net/vlu-net_arch.py†L262-L371】

### 34) DetailRefinerNet
* **Key features:** EnhancedRefinementBlock with GELU and SE channel attention, grouped residual stacks, and long-range feature fusion via concatenation and 1×1 conv.【F:arches/detailrefinernet_arch.py†L8-L33】【F:arches/detailrefinernet_arch.py†L36-L92】
* **Operating principle:** Initial conv → sequential ERB groups → concatenate group outputs → fusion conv → final conv → global residual add.【F:arches/detailrefinernet_arch.py†L55-L101】
* **Speed/compute:** Pure CNN with small 3×3/1×1 convs; typically faster than attention/SSM models at similar depth.【F:arches/detailrefinernet_arch.py†L36-L101】
* **Size/memory:** Moderate; parameters scale with num_groups and num_blocks_per_group, plus fusion conv for concatenated features.【F:arches/detailrefinernet_arch.py†L55-L92】
* **Textures:** SE attention reweights channels for detail enhancement; local residual blocks preserve edges and textures.【F:arches/detailrefinernet_arch.py†L8-L52】
* **Advantages:** Simple, efficient refinement network with channel attention and long-range fusion; easy to deploy.【F:arches/detailrefinernet_arch.py†L55-L101】
* **Cons:** Lacks explicit global attention; texture modeling is local and depends on depth/width scaling.【F:arches/detailrefinernet_arch.py†L36-L101】
* **Schematic:** Input → initial conv → ERB groups → concat + fusion → final conv → output + residual.【F:arches/detailrefinernet_arch.py†L55-L101】

### 35) EMT
* **Key features:** MixedTransformerBlock stacks combining SWSA window attention and TokenMixer blocks, Swish activations, and PixelShuffle-based upsampling tail.【F:arches/emt_arch.py†L340-L420】【F:arches/emt_arch.py†L520-L660】【F:arches/emt_arch.py†L568-L646】
* **Operating principle:** MeanShift normalization → head conv → mixed transformer body with residual → Upsampler tail → add mean back.【F:arches/emt_arch.py†L568-L646】
* **Speed/compute:** Window attention (SWSA) plus token mixing is heavier than pure CNN SR but cheaper than full global attention; compute scales with n_blocks × n_layers.【F:arches/emt_arch.py†L340-L420】【F:arches/emt_arch.py†L520-L660】
* **Size/memory:** Moderate-to-high; attention projections and multiple transformer layers per block increase activations and parameters.【F:arches/emt_arch.py†L520-L646】
* **Textures:** Window attention and token mixing capture local structures and repeated textures while maintaining efficient spatial mixing.【F:arches/emt_arch.py†L340-L420】【F:arches/emt_arch.py†L520-L660】
* **Advantages:** Balanced transformer SR with efficient windowed attention and flexible upsampling options.【F:arches/emt_arch.py†L520-L646】
* **Cons:** Still heavier than compact CNNs; window attention limits very long-range context without deeper stacks.【F:arches/emt_arch.py†L340-L420】【F:arches/emt_arch.py†L568-L646】
* **Schematic:** Input → MeanShift → head conv → MixedTransformerBlocks → residual → Upsampler → output.【F:arches/emt_arch.py†L568-L646】

### 36) ParagonSR
* **Key features:** ParagonBlock combines multi-scale InceptionDWConv2d, GatedFFN, and reparameterizable ReparamConvV2; residual groups with LayerScale stabilize deep stacks.【F:arches/paragonsr_arch.py†L122-L206】【F:arches/paragonsr_arch.py†L208-L239】
* **Operating principle:** conv_in → ResidualGroup stack → conv_fuse + residual → MagicKernelSharp2021 upsample → conv_out.【F:arches/paragonsr_arch.py†L245-L320】
* **Speed/compute:** ReparamConvV2 can fuse branches for eval, reducing runtime overhead versus multi-branch training-time convs; overall cost scales with groups/blocks.【F:arches/paragonsr_arch.py†L62-L120】【F:arches/paragonsr_arch.py†L245-L320】
* **Size/memory:** Scales with num_groups and num_blocks; multi-branch convs add parameters at training time but can be fused for deployment.【F:arches/paragonsr_arch.py†L62-L120】【F:arches/paragonsr_arch.py†L245-L320】
* **Textures:** InceptionDWConv2d captures multi-scale spatial detail, while GatedFFN enhances texture transformation capacity.【F:arches/paragonsr_arch.py†L122-L170】
* **Advantages:** High-quality CNN-first SR with deployable reparameterization and strong texture modeling.【F:arches/paragonsr_arch.py†L62-L120】【F:arches/paragonsr_arch.py†L245-L320】
* **Cons:** More complex than plain residual CNNs; multi-branch training costs more compute/memory before fusion.【F:arches/paragonsr_arch.py†L62-L120】
* **Schematic:** Input → conv_in → ResidualGroups → conv_fuse + residual → MagicKernelSharp upsample → conv_out.【F:arches/paragonsr_arch.py†L245-L320】

### 37) ParagonSR2
* **Key features:** Dual-path design: fixed MagicKernelSharp2021 base upsampler plus learned detail path; variants select lightweight conv blocks or window-attention photo/pro blocks; detail gain is learnable.【F:arches/paragonsr2_arch.py†L39-L129】【F:arches/paragonsr2_arch.py†L1171-L1308】
* **Operating principle:** Base upsample → conv_in → residual groups (variant-dependent) → pixelshuffle detail → detail_gain → base + detail output.【F:arches/paragonsr2_arch.py†L1171-L1365】
* **Speed/compute:** Base path is cheap; compute dominated by selected variant (realtime/stream conv vs photo/pro window attention), with optional checkpointing for memory trade-offs.【F:arches/paragonsr2_arch.py†L1171-L1308】
* **Size/memory:** Scales with num_groups × num_blocks and variant; attention variants add window attention parameters and buffers (relative positions/masks).【F:arches/paragonsr2_arch.py†L1171-L1308】
* **Textures:** Base reconstruction provides stable low-frequency content; learned detail path refines high-frequency textures and can leverage attention in photo/pro variants.【F:arches/paragonsr2_arch.py†L1171-L1365】
* **Advantages:** Deployment-friendly dual-path SR with configurable speed/quality profiles and stable base reconstruction.【F:arches/paragonsr2_arch.py†L1171-L1308】
* **Cons:** Attention variants are heavier; quality relies on balancing base/detail gain and variant-specific hyperparameters.【F:arches/paragonsr2_arch.py†L1171-L1308】
* **Schematic:** Input → MagicKernelSharp base → conv_in → residual body → pixelshuffle detail → base + detail_gain × detail → output.【F:arches/paragonsr2_arch.py†L1171-L1365】

### 38) ElysiumSR
* **Key features:** Pure residual CNN with configurable ResidualBlock depth, DropPath stochastic depth, and PixelShuffle upsampling; multiple size variants share a common core.【F:arches/elysiumsr_arch.py†L55-L151】【F:arches/elysiumsr_arch.py†L169-L236】
* **Operating principle:** conv_in → residual block stack (+ DropPath) → conv_fuse + residual → PixelShuffle upsampler → conv_out.【F:arches/elysiumsr_arch.py†L169-L236】
* **Speed/compute:** Convolution-only design is faster than attention/SSM models; compute scales linearly with num_blocks and num_feat.【F:arches/elysiumsr_arch.py†L169-L236】
* **Size/memory:** Scales with num_feat/num_blocks; no attention buffers, so memory is predictable and relatively light.【F:arches/elysiumsr_arch.py†L169-L236】
* **Textures:** Residual blocks preserve local textures but lack explicit global context; DropPath can improve training robustness.【F:arches/elysiumsr_arch.py†L55-L151】
* **Advantages:** Simple, stable, deployable CNN SR with multiple capacity tiers.【F:arches/elysiumsr_arch.py†L55-L151】【F:arches/elysiumsr_arch.py†L169-L236】
* **Cons:** Limited global modeling compared to transformer/SSM approaches; texture refinement depends on depth/width scaling.【F:arches/elysiumsr_arch.py†L169-L236】
* **Schematic:** Input → conv_in → residual blocks → conv_fuse + residual → PixelShuffle → conv_out.【F:arches/elysiumsr_arch.py†L169-L236】

### 39) DIS
* **Key features:** Minimal SR model with FastResBlock or depthwise LightBlock options, PixelShuffle upsampling, and global residual learning via bilinear base upsampling.【F:arches/dis_arch.py†L21-L120】【F:arches/dis_arch.py†L122-L206】
* **Operating principle:** Head conv → lightweight residual body → fusion conv + residual → PixelShuffle upsampler → tail conv → add bilinear base.【F:arches/dis_arch.py†L137-L206】
* **Speed/compute:** Designed for speed with minimal depth, no attention, and optional depthwise separable convs; typically faster than transformer SRs.【F:arches/dis_arch.py†L12-L120】【F:arches/dis_arch.py†L122-L206】
* **Size/memory:** Small parameter count; memory dominated by feature maps rather than attention buffers.【F:arches/dis_arch.py†L137-L206】
* **Textures:** Local conv-only modeling, so texture recovery is limited to local receptive fields; global residual preserves coarse structure.【F:arches/dis_arch.py†L137-L206】
* **Advantages:** Very fast and simple SR baseline; easy to deploy and scale to real-time use cases.【F:arches/dis_arch.py†L70-L120】【F:arches/dis_arch.py†L122-L206】
* **Cons:** Limited global context and lower detail capacity versus attention/SSM-based SR models.【F:arches/dis_arch.py†L137-L206】
* **Schematic:** Input → head conv → FastRes/Light blocks → fusion → PixelShuffle upsample → tail conv → output + bilinear base.【F:arches/dis_arch.py†L137-L206】

### 40) catanet
* **Key features:** Alternates global TAB (token aggregation with IRCA/IASA attention) and local LRSA blocks, with patch-based attention windows and ConvFFN mixing.【F:arches/catanet_arch.py†L113-L233】【F:arches/catanet_arch.py†L332-L389】【F:arches/catanet_arch.py†L440-L487】
* **Operating principle:** First conv → repeated [TAB + LRSA + mid conv] residual blocks → PixelShuffle upsampling → final conv + global bilinear residual.【F:arches/catanet_arch.py†L440-L527】
* **Speed/compute:** Attention-heavy due to token grouping and scaled dot-product attention; compute depends on patch_size, num_tokens, and group_size schedules.【F:arches/catanet_arch.py†L113-L233】【F:arches/catanet_arch.py†L440-L487】
* **Size/memory:** Moderate-to-high; maintains token statistics (means buffers) and attention projections across multiple blocks and heads.【F:arches/catanet_arch.py†L196-L233】【F:arches/catanet_arch.py†L440-L487】
* **Textures:** Global token aggregation (TAB) plus local patch attention (LRSA) improves texture/detail recovery across scales and patterns.【F:arches/catanet_arch.py†L113-L233】【F:arches/catanet_arch.py†L332-L389】
* **Advantages:** Combines global token context with local patch attention, offering strong texture/detail modeling for SR tasks.【F:arches/catanet_arch.py†L113-L233】【F:arches/catanet_arch.py†L440-L487】
* **Cons:** Heavier than CNN baselines; attention hyperparameters (patch sizes, tokens) require tuning for speed/memory balance.【F:arches/catanet_arch.py†L440-L487】
* **Schematic:** Input → conv → [TAB → LRSA → mid conv]×N → PixelShuffle upsample → final conv → output + bilinear base.【F:arches/catanet_arch.py†L440-L527】
