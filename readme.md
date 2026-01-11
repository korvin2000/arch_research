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
