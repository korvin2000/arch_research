# LLM-Ready Best-Practices Guide for Image Restoration + SISR Architectures

::meta::
doc_kind: best_practices
primary_domain: image_restoration_and_sisr
target_audience: LLM_and_engineers
source_context: readme.md, overview.md, ideas/idea-05.md, ideas/idea-08.md
modeling_focus: pytorch
format: machine_readable_markdown
version: 0.1
::end::

## 0. Scope, positioning, and constraints (from idea-05/idea-08)
::scope::
tasks: [image_restoration, single_image_super_resolution]
priorities: [stability, efficiency, texture_fidelity, scalability]
constraints:
  - no_full_global_attention: true
  - prefer_windowed_attention: true
  - prefer_reparameterizable_convs: true
  - allow_selective_routing_or_moe: true
  - allow_frequency_decomposition: true
::end::

### 0.1 Behavior-defining invariants
- invariant: preserve spatial resolution inside blocks; only change H/W at explicit down/up stages.
- invariant: residual connections must align shapes and dtypes; avoid implicit broadcasting.
- invariant: routing and MoE should not touch the low-frequency stability path by default.
- invariant: any FFT block must keep complex->real conversion stable and deterministic.
- invariant: reparameterization must be lossless (verify with numerical tolerance).
- invariant: if padding for window attention, crop back exactly to original size.

### 0.2 Failure modes checklist (non-exhaustive)
- failure: expert collapse in MoE → mitigate with load-balancing loss and warmup.
- failure: window attention boundary seams → use shift/overlap or consistent padding.
- failure: FFT ringing → clamp/regularize frequency amplification; prefer gated fusion.
- failure: large-kernel slowdowns → reparameterize or use sparse/partial large kernels.
- failure: dysample offsets unstable → initialize scope to 0 and constrain magnitude.
- failure: kernel-aware blocks overfit → regularize kernel embedding and limit kernel_size.

## 1. LLM-Readable Requirements Map
::requirements_map::
hybrid_backbone:
  must_include: [cnn_stem, gated_conv_blocks, window_attention_mid_or_low_res]
frequency_handling:
  must_include: [low_high_split, optional_fourier_unit, cross_frequency_fusion]
conditional_compute:
  must_include: [token_or_expert_routing, optional_early_exit]
upsample_head:
  allowed: [pixelshuffle, pixelshuffledirect, nearest_plus_conv, dysample]
deployment:
  must_include: [reparameterization_path, shape_checks, inference_mode_switch]
::end::

## 2. Architectural Patterns (LLM-Ready Blocks)

### 2.1 Block Template: Frequency-Split Hybrid Block
::block::
name: freq_split_hybrid_block
input: [B, C, H, W]
output: [B, C, H, W]
substeps:
  - low = LowPass(x)
  - high = x - Up(low)
  - low = GatedCNN(low)
  - high = WindowAttention(high)
  - high = MoE(high)  # optional
  - fused = Fuse(low, high)  # concat+1x1 or cross-attn
  - fused = ReparamLargeKernel(fused)  # optional
  - return fused + x
invariants:
  - keep dtype/device consistent across branches
  - ensure low/high are same shape before fuse
risk:
  - high-frequency dominance causes artifacts → add gating or channel attention
::end::

### 2.2 Block Template: Gated CNN Block (Conv-first)
::block::
name: gated_cnn_block
input: [B, C, H, W]
output: [B, C, H, W]
steps:
  - norm(x)
  - split = conv1(x) -> (g, i, c)
  - c = depthwise_inception(c)
  - out = act(conv2(act(g) * concat(i, c)))
  - return out * gamma + x
notes:
  - gamma starts small to stabilize deep stacks
  - prefer Mish/GELU for smooth gradients
::end::

### 2.3 Block Template: Window Attention with Relative Bias
::block::
name: window_attention_relative_bias
input: [B, N, C] or [B, H, W, C]
output: same_shape
steps:
  - partition windows
  - compute qkv and scaled dot-product
  - add relative position bias
  - apply softmax and projection
  - reverse windows and crop to original size
invariants:
  - window size divides padded H/W
  - mask applied when shifted windows are used
::end::

### 2.4 Block Template: DySample Upsampler
::block::
name: dysample_upsampler
input: [B, C, H, W]
output: [B, C_out, H*s, W*s]
steps:
  - offset = conv(x) * scope(x).sigmoid() * 0.5 + init_pos
  - pixel_shuffle offsets and grid_sample
  - optional end_conv to project to out_ch
risks:
  - unstable offsets if scope init not zero
  - grid_sample precision issues in FP16 → prefer autocast or fp32 for offsets
::end::

### 2.5 Block Template: Kernel-Aware Attention
::block::
name: kernel_aware_attention
input: [B, C, H, W], kernel_map: [B, K^2, H, W]
output: [B, C, H, W]
steps:
  - x_feat = conv(x)
  - k_feat = conv(kernel_map)
  - att = sigmoid(conv(cat(x_feat, k_feat)))
  - out = x_feat * att + x
invariants:
  - kernel_map must match spatial size of x
  - kernel_map channel dim must be K^2
::end::

## 3. Reference Implementations: Critically-Reviewed Architectures

### 3.1 Architecture Reference: PLKSR / RealPLKSR
::arch_ref::
name: PLKSR / RealPLKSR
source_file: plksr/realplksr_arch.py
focus:
  - partial_large_kernel
  - elementwise_attention
  - group_norm
  - pixelshuffle_or_dysample
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class PLKConv2d(nn.Module):
    """Partial Large Kernel Convolutional Layer"""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    """Element-wise Attention"""

    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ):
        super().__init__()

        # Local Texture
        self.channel_mixer = DCCM(dim)

        # Long-range Dependency
        pdim = int(dim * split_ratio)

        # Conv Layer
        self.lk = PLKConv2d(pdim, kernel_size)

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)

        # Group Normalization
        self.norm = nn.GroupNorm(norm_groups, dim)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)

        return x + x_skip


@ARCH_REGISTRY.register()
class realplksr(nn.Module):
    """Partial Large Kernel CNNs for Efficient Super-Resolution:
    https://arxiv.org/abs/2404.11848
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = upscale,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
        dropout: float = 0,
        dysample: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.upscale = upscaling_factor
        self.dysample = dysample
        if not self.training:
            dropout = 0

        self.feats = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [
                PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)
                for _ in range(n_blocks)
            ]
            + [nn.Dropout2d(dropout)]
            + [nn.Conv2d(dim, out_ch * upscaling_factor**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=upscaling_factor**2, dim=1
        )

        if dysample and upscaling_factor != 1:
            groups = out_ch if upscaling_factor % 2 != 0 else 4
            self.to_img = DySample(
                in_ch * upscaling_factor**2,
                out_ch,
                upscaling_factor,
                groups=groups,
                end_convolution=True if upscaling_factor != 1 else False,
            )
        else:
            self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + self.repeat_op(x)
        if not self.dysample or (self.dysample and self.upscale != 1):
            x = self.to_img(x)
        return x
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.2 Architecture Reference: GFISRV2
::arch_ref::
name: GFISRV2
source_file: gfisrv2/gfisrv2_arch.py
focus:
  - dysample_offsets
  - layer_norm_channels_last
  - adaptive_upsample
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=1,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.3 Architecture Reference: GateRV3
::arch_ref::
name: GateRV3
source_file: gaterv3/gaterv3_arch.py
focus:
  - reparameterizable_conv
  - gated_paths
  - inception_dwconv
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain1: int = 1, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.bias = bias
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        nn.init.trunc_normal_(self.sk.weight, std=0.02)
        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False  # pyright: ignore[reportOptionalMemberAccess]
            self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportOptionalMemberAccess]

        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat
        if self.bias:
            self.bias_concat = self.bias_concat + sk_b  # pyright: ignore[reportOperatorIssue,reportPossiblyUnboundVariable]
            self.eval_conv.bias.data = self.bias_concat  # pyright: ignore[reportOptionalMemberAccess]

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)
        return out

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.4 Architecture Reference: SpanPP / SpanC
::arch_ref::
name: SpanPP / SpanC
source_file: spanpp/spanpp_arch.py
focus:
  - reparameterization
  - repconv_fusion
  - implicit_kernel
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class RepConv(nn.Module):
    def __init__(self, in_dim=3, out_dim=32) -> None:
        super().__init__()
        self.conv1 = SeqConv3x3(in_dim, out_dim, 2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv3 = Conv3XC(in_dim, out_dim)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.alpha = nn.Parameter(torch.randn(3), requires_grad=True)
        # self.alpha.register_hook(lambda grad: grad * 100)
        self.forward_module = self.train_forward

        nn.init.constant_(self.alpha, 1.0)

    def fuse(self) -> None:
        conv1_w, conv1_b = self.conv1.rep_params()
        conv2_w, conv2_b = self.conv2.weight, self.conv2.bias
        self.conv3.update_params()
        conv3_w, conv3_b = self.conv3.eval_conv.weight, self.conv3.eval_conv.bias
        device = self.conv_3x3_rep.weight.device
        sum_weight = (
            self.alpha[0] * conv1_w + self.alpha[1] * conv2_w + self.alpha[2] * conv3_w
        ).to(device)
        sum_bias = (
            self.alpha[0] * conv1_b + self.alpha[1] * conv2_b + self.alpha[2] * conv3_b
        ).to(device)
        self.conv_3x3_rep.weight = nn.Parameter(sum_weight)
        self.conv_3x3_rep.bias = nn.Parameter(sum_bias)

    def train_forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.alpha[0] * x1 + self.alpha[1] * x2 + self.alpha[2] * x3

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.fuse()
        return self

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.conv_3x3_rep(x)

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.5 Architecture Reference: SeeMore (MoE + StripedConvFormer)
::arch_ref::
name: SeeMore (MoE + StripedConvFormer)
source_file: SeeMore/seemore_arch.py
focus:
  - moe_blocks
  - striped_conv_attention
  - gated_ffn
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
@ARCH_REGISTRY.register()
class SeemoRe(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_experts: int = 6,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = False,
                 global_kernel_size: int = 11,
                 recursive: int = 2,
                 lr_space: int = 1,
                 topk: int = 2,):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        
        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)
        
        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [ResGroup(in_ch=embedding_dim, 
                       num_experts=num_experts, 
                       use_shuffle=use_shuffle,
                       topk=topk,
                       lr_space=lr_space,
                       recursive=recursive,
                       global_kernel_size=global_kernel_size) for i in range(num_layers)]
        )
        
        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale**2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x
        
        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)
                
        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x
    
    
    
#############################
# Components
#############################    
class ResGroup(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 global_kernel_size: int = 11,
                 lr_space: int = 1,
                 topk: int = 2,
                 recursive: int = 2,
                 use_shuffle: bool = False):
        super().__init__()
        
        self.local_block = RME(in_ch=in_ch, 
                               num_experts=num_experts, 
                               use_shuffle=use_shuffle, 
                               lr_space=lr_space, 
                               topk=topk, 
                               recursive=recursive)
        self.global_block = SME(in_ch=in_ch, 
                                kernel_size=global_kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_block(x)
        x = self.global_block(x)
        return x

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.6 Architecture Reference: SeeMore (MoE block internals)
::arch_ref::
name: SeeMore (MoE block internals)
source_file: SeeMore/seemore_arch.py
focus:
  - moe_routing
  - local_experts
  - aggregate_conv
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 use_shuffle: bool = False,
                 lr_space: str = "linear",
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2*in_ch, kernel_size=1, padding=0)
        )
        
        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )
        
        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True),
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.7 Architecture Reference: MicroSR
::arch_ref::
name: MicroSR
source_file: team07_MicroSR/MicroSR_Model.py
focus:
  - window_attention
  - patch_embed
  - pixelshuffle_upsample
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class MicroSR(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 gc = 32,
                 **kwargs):
        super(MicroSR, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            layer = RDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                                 num_heads= num_heads[i_layer], window_size=window_size, depth=0,
                                 shift_size= window_size//2, mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,  drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                                             norm_layer=norm_layer,gc=gc, img_size=img_size, patch_size=patch_size)

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.8 Architecture Reference: ParagonSR2
::arch_ref::
name: ParagonSR2
source_file: arches/paragonsr2_arch.py
focus:
  - classical_base_upsampler
  - rmsnorm_layerscale
  - window_attention
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
# ============================================================================
# 1. CLASSICAL BASE UPSAMPLER (MAGIC KERNEL SHARP 2021)
# ============================================================================


def get_magic_kernel_weights():
    """
    Low-pass reconstruction kernel from Costella's Magic Kernel.
    """
    return torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])


def get_magic_sharp_2021_kernel_weights():
    """Returns the weights for the Magic Kernel Sharp 2021 (MKS21) 3-tap filter."""
    # Weights: [-0.015, 0.138, -0.015] -> [0.138, 0.862, 0.138] unnormalized?
    # Actually MKS2021 is usually approximating Lanczos3 or Catmull-Rom.
    # Here we use a standard sharp bicubic-like approximation.
    return torch.tensor([-1 / 32, 0, 9 / 32, 16 / 32, 9 / 32, 0, -1 / 32])


class SeparableConv(nn.Module):
    """
    Fixed separable convolution for classical reconstruction kernels.

    These filters are frozen by design and must never be trained.
    """

    def __init__(self, channels: int, kernel: torch.Tensor) -> None:
        super().__init__()
        k = len(kernel)
        self.register_buffer("kernel", kernel)

        self.h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=channels,
            bias=False,
        )
        self.v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            groups=channels,
            bias=False,
        )

        with torch.no_grad():
            self.h.weight.copy_(kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1))
            self.v.weight.copy_(kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1))

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.h(x))


class MagicKernelSharp2021Upsample(nn.Module):
    """
    Fixed classical upsampler based on Magic Kernel Sharp 2021.

    Provides a strong low-frequency base reconstruction that the neural
    network refines with learned high-frequency detail.
    """

    def __init__(self, in_ch: int, scale: int, alpha: float) -> None:
        super().__init__()
        self.scale = scale
        self.alpha = alpha

        self.sharp = SeparableConv(in_ch, get_magic_sharp_2021_kernel_weights())
        self.blur = SeparableConv(in_ch, get_magic_kernel_weights())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional pre-sharpening
        if self.alpha > 0:
            x = x + self.alpha * (self.sharp(x) - x)

        # Nearest-neighbor upsampling
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        # Reconstruction blur
        return self.blur(x)


# ============================================================================
# 2. NORMALIZATION & RESIDUAL SCALING
# ============================================================================


class RMSNorm(nn.Module):
    """
    Spatial RMS Normalization.

    More stable than BatchNorm for SR and safe in FP16.
    Computes variance in FP32 for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate variance in FP32 for stability, then cast back
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=1, keepdim=True)
        rms = torch.sqrt(variance + self.eps).to(x.dtype)
        return self.scale * x / rms + self.bias


class LayerScale(nn.Module):
    """
    LayerScale: Learnable per-channel scaling factor initialized to a small value.
    Crucial for stabilizing deep networks, especially in FP16.
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.9 Architecture Reference: ESC Real
::arch_ref::
name: ESC Real
source_file: ESC/esc_real_arch.py
focus:
  - conv_attention
  - flex_attention
  - rpe_bias
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int):
        super().__init__()
        self.pdim = pdim
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)

            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + x1_

            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=13 // 2) \
                               + F.conv2d(x[:, :self.pdim], dynamic_kernel, stride=1, padding=self.sk_size // 2,
                                          groups=self.pdim)

            # For Mobile Conversion, uncomment the following code
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(x_1, dynamic_kernel, stride=1, padding=1, groups=16)
            # x = torch.cat([x_1, x_2], dim=1)
        return x

    def extra_repr(self):
        return f'pdim={self.pdim}'


class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.10 Architecture Reference: SFHformer
::arch_ref::
name: SFHformer
source_file: SFHformer/SFHformer.py
focus:
  - fourier_unit
  - local_global_mixers
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class FourierUnit(nn.Module):
    # simple tasks, e.g. dehazing\deraining can set groups=1 for better latency; complex tasks, e.g. motion blur can set groups=4 for better performance. 
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
             nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
             nn.Softmax(dim=1)
        )

        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                                        padding=1, stride=1, groups=in_channels * 2,bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2*c, h, -1)  # (batch, c*2, h, w/2+1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class TokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)

        return x
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.11 Architecture Reference: DiMoSR
::arch_ref::
name: DiMoSR
source_file: DiMoSR/dimosr_arch.py
focus:
  - multi_stage_blocks
  - feature_fusion
  - pixelshuffle_head
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
@ARCH_REGISTRY.register()
class DiMoSR(nn.Module):    
    def __init__(self, scale=4, num_feat=36, num_block=18, **kwargs):
        super(DiMoSR, self).__init__()
        # Define your architecture components
        self.shallow_conv = nn.Conv2d(3, num_feat, kernel_size=3, padding=1)
            
        self.stage1_blocks = nn.Sequential(*[ResBottleneck(num_feat) for _ in range(num_block//3)])
        self.stage2_blocks = nn.Sequential(*[ResBottleneck(num_feat) for _ in range(num_block//3)])
        remanin = num_block - 2*(num_block//3)
        self.stage3_blocks = nn.Sequential(*[ResBottleneck(num_feat) for _ in range(remanin)])

        self.fusion = nn.Conv2d(num_feat*3, num_feat, kernel_size=1)
        # Upsampling layer: increases spatial resolution using PixelShuffle.
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, (scale ** 2) * 3, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # Implement the forward pass
        shallow_features = self.shallow_conv(x)
        stage1_features = self.stage1_blocks(shallow_features)
        stage1_out = stage1_features + shallow_features
        
        # Stage 2: Refinement with knowledge from stage 1
        stage2_features = self.stage2_blocks(stage1_out)
        stage2_out = stage2_features + stage1_out

        stage3_features = self.stage3_blocks(stage2_out)
        stage3_out = stage3_features + stage2_out
        
        # Concatenate and fuse multi-stage features for richer representation
        concat_features = torch.cat([stage1_out, stage2_out, stage3_out], dim=1)
        fused_features = self.fusion(concat_features)
        
        # Generate the high-resolution output
        out = self.upsampler(fused_features)
        return out
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.12 Architecture Reference: DiMoSR ResBottleneck
::arch_ref::
name: DiMoSR ResBottleneck
source_file: DiMoSR/dimosr_arch.py
focus:
  - dilated_conv_paths
  - scale_bias_attention
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class ResBottleneck(nn.Module):
    def __init__(self, num_feat):
        super(ResBottleneck, self).__init__()

        self.norm1 = LayerNorm(num_feat)
        self.norm2 = LayerNorm(num_feat)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//2, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//2, num_feat//2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//2, num_feat//2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//2, num_feat, kernel_size=1, padding=0),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=8, dilation=8),
            nn.SiLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=12, dilation=12),
            nn.SiLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=16, dilation=16),
            nn.SiLU(inplace=True),
        )

        self.agg = nn.Conv2d((num_feat//4)*4, num_feat*3, kernel_size=1, padding=0)
        self.integrate = nn.Conv2d(num_feat*2, num_feat, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        scale, bias, attn = self.agg(torch.cat([c1, c2, c3, c4], dim=1)).chunk(3, 1)
        attn = torch.sigmoid(attn)

        out1 = x * scale + bias
        out2 = x * attn
        
        x = self.integrate(torch.cat([out1, out2], dim=1)) + identity
        
        out = self.bottleneck(self.norm2(x))
        out = out + x

        return out
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.13 Architecture Reference: LoFormer Flow (KernelPrior)
::arch_ref::
name: LoFormer Flow (KernelPrior)
source_file: LoFormer/Flow_arch.py
focus:
  - normalizing_flow_kernel_prior
  - log_prob_for_kernels
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class KernelPrior(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, kernel_size=0, alpha=1e-6, normalization=1,
                 cond_label_size=None, batch_norm=True):
        super().__init__()

        # parameters of kernel pre-processing
        self.register_buffer('kernel_size', torch.ones(1)*kernel_size)
        self.register_buffer('alpha', torch.ones(1)*alpha)
        self.register_buffer('normalization', torch.ones(1)*normalization)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask  # like permutation, though a waste of parameters in the first layer
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        # log_prob(u) is always negative, sum_log_abs_det_jacobians mostly negative -> log_prob is always negative
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return self.base_dist.log_prob(u).sum(1) + sum_log_abs_det_jacobians, u  # should all be summation

    def post_process(self, x):
        # inverse process of pre_process in dataloader
        # self._clamp_abs(self.alpha.data, 1e-3)
        x = x.view(x.shape[0], 1, int(self.kernel_size), int(self.kernel_size))
        x = ((torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha))
        x = x * self.normalization
        return x
    # def _clamp_abs(self, data, value):
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.14 Architecture Reference: LoFormer Flow (Coupling)
::arch_ref::
name: LoFormer Flow (Coupling)
source_file: LoFormer/Flow_arch.py
focus:
  - masked_coupling
  - invertible_transform
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class LinearMaskedCoupling(nn.Module):
    """ Coupling Layers """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        # stored in state_dict, but not trained & not returned by nn.parameters(); similar purpose as nn.Parameter objects
        # this is because tensors won't be saved in state_dict and won't be pushed to the device
        self.register_buffer('mask', mask)  # 0,1,0,1

        # scale function
        # for conditional version, just concat label as the input into the network (conditional way of SRMD)
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]

        self.s_net = nn.Sequential(*s_net)

        # translation function, the same structure
        self.t_net = copy.deepcopy(self.s_net)

        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        log_s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(
            -log_s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = (- (1 - self.mask) * log_s).sum(
            1)  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        log_s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))  # log of scale, log(s)
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))  # translation, t
        x = mu + (1 - self.mask) * (u * log_s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = ((1 - self.mask) * log_s).sum(1)  # log det dx/du

        return x, log_abs_det_jacobian
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.15 Architecture Reference: LoFormer BaseBlock
::arch_ref::
name: LoFormer BaseBlock
source_file: LoFormer/baseblock.py
focus:
  - kernel_attention
  - fft_deblur_filtering
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
def deblurfeaturefilter_fft(image, kernel, NSR=None):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    # otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-7)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.16 Architecture Reference: UFPNet (NAF + kernel-aware)
::arch_ref::
name: UFPNet (NAF + kernel-aware)
source_file: AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py
focus:
  - naf_block
  - simple_gate
  - kernel_attention
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.17 Architecture Reference: UFPNet (NAFBlock_kernel)
::arch_ref::
name: UFPNet (NAFBlock_kernel)
source_file: AdaRevD/UFPNet/UFPNet_code_uncertainty_arch.py
focus:
  - kernel_conditioned_block
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=21):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, inp, kernel):
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.18 Architecture Reference: RHA (DySample + UniUpsample)
::arch_ref::
name: RHA (DySample + UniUpsample)
source_file: rha/arch.py
focus:
  - dysample_upsample
  - multi_mode_upsampler
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = 'Incorrect in_channels and groups values.'
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij'))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device, pin_memory=True).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,  # Only pixelshuffle
        group: int = 4,  # Only DySample
    ) -> None:
        m = []

        if scale == 1 or upsample == 'conv':
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == 'pixelshuffledirect':
            m.extend([nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)])
        elif upsample == 'pixelshuffle':
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)])
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == 'nearest+conv':
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == 'dysample':
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
                dys_dim = mid_dim
            else:
                dys_dim = in_dim
            m.append(DySample(dys_dim, out_dim, scale, group))
        else:
            raise ValueError(f'An invalid Upsample was selected. Please choose one of {SampleMods}')
        super().__init__(*m)

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.19 Architecture Reference: RHA (FocusedLinearAttention)
::arch_ref::
name: RHA (FocusedLinearAttention)
source_file: rha/arch.py
focus:
  - focused_linear_attention
  - windowed_attention
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class FocusedLinearAttention(nn.Module):
    r"""https://github.com/LeapLabTHU/FLatten-Transformer/blob/master/models/flatten_swin.py
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""

    def __init__(
        self,
        dim: int = 64,
        window_size: int = 8,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        focusing_factor: int = 3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(
            in_channels=head_dim,
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.20 Architecture Reference: StyleGAN2 (Generator core)
::arch_ref::
name: StyleGAN2 (Generator core)
source_file: others/stylegan2_arch.py
focus:
  - style_mlp
  - modulated_convs
  - noise_buffers
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


@ARCH_REGISTRY.register()
class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1):
        super(StyleGAN2Generator, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp,
                    activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'],
            channels['4'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False, resample_kernel=resample_kernel)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample',
                    resample_kernel=resample_kernel,
                ))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.21 Architecture Reference: CFAT (Window Attention + CAB)
::arch_ref::
name: CFAT (Window Attention + CAB)
source_file: CFAT/cfat.py
focus:
  - cab
  - window_attention
  - relative_bias
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):   #[1, 180, 64, 64]
        return self.cab(x)
###############------Channel_Attention_Block(CAB)------###############



###############------Multi_Layer_Perceptron------###############
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
###############------Multi_Layer_Perceptron------###############



###################---------Window_Attention_D---------###################
class WindowAttention_D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        
        ##Arguments
        self.dim = dim   #180
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  #6
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        ##Module
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, rpi, mask=None):  #[16, 16*16, 180]
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape  #[16, 256, 180]
        
        ##########------q, k, v------##########
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4) #[16, 256, 540]->[3, 16, 6, 256, 30]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) #[16, 6, 256, 30]
        ##########------q, k, v------##########
        
        ##########------q*k------##########
        q = q * self.scale  #scale=0.18257418583505536 #[16, 6, 256, 30]
        attn = (q @ k.transpose(-2, -1))  #[16, 6, 256, 256]
        ##########------q*k------##########
        
        ##########--------Relative_Position_Bias--------##########
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
                                                                                       # Wh*Ww,Wh*Ww,nH #[256, 256, 6]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww #[6, 256, 256]
        attn = attn + relative_position_bias.unsqueeze(0)  #[16, 6, 256, 256]+[1, 6, 256, 256]=[16, 6, 256, 256]
        ##########--------Relative_Position_Bias--------##########
        
        ##########------masking+softmax------##########
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)   #[32, 6, 256, 256] [16, 256, 256]
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        ##########------masking+softmax------##########

        attn = self.attn_drop(attn)  #[16, 6, 256, 256]
        
        ##########------(qk)*v+Linear------##########
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)  #[16, 6, 256, 256]*[16, 6, 256, 30]=[16, 6, 256, 30]->[16, 256, 180]
        x = self.proj(x)  #[16, 256, 180]
        ##########------(qk)*v+Linear------##########
        
        x = self.proj_drop(x)  #[16, 256, 180]
        
        return x
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.22 Architecture Reference: HIT-SRF
::arch_ref::
name: HIT-SRF
source_file: neosr/hitsrf_arch.py
focus:
  - hierarchical_transformer
  - upsample_modes
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
        #####################################################################################################
        # 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        # 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual Hierarchical Transformer blocks (RHTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                base_win_size=base_win_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                value_drop=value_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hier_win_ratios=hier_win_ratios,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        #####################################################################################################
        # 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale,
                embed_dim,
                num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.23 Architecture Reference: MoESR
::arch_ref::
name: MoESR
source_file: moesr/arch.py
focus:
  - gated_cnn_block
  - msg_path
  - pixel_unshuffle
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = InceptionDWConv2d(conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return (x * self.gamma) + shortcut


class MSG(nn.Module):
    def __init__(self, dim, expansion_msg=1.5):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.PixelUnshuffle(2), nn.LeakyReLU(0.1, True))
        self.gated = nn.Sequential(*[GatedCNNBlock(dim, expansion_ratio=expansion_msg) for _ in range(3)])
        self.up = nn.Sequential(nn.Conv2d(dim, dim * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

    def forward(self, x):
        out = self.down(x)
        out = self.gated(out)
        return self.up(out) + x

```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

### 3.24 Architecture Reference: MoSRv2
::arch_ref::
name: MoSRv2
source_file: mosrv2/arch.py
focus:
  - dysample
  - multi_mode_upsample
key_takeaways:
  - prefer stable residual paths and explicit normalization
  - ensure that operator choices match deployment targets
  - match module to frequency/texture duty (low vs high)
::end::

```python
class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = 'Incorrect in_channels and groups values.'
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij'))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device, pin_memory=True).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,  # Only pixelshuffle and DySample
        group: int = 4,  # Only DySample
    ) -> None:
        m = []

        if scale == 1 or upsample == 'conv':
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == 'pixelshuffledirect':
            m.extend([nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)])
        elif upsample == 'pixelshuffle':
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)])
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == 'nearest+conv':
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == 'dysample':
            if mid_dim != in_dim:
                m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
                dys_dim = mid_dim
            else:
                dys_dim = in_dim
            m.append(DySample(dys_dim, out_dim, scale, group))
        else:
            raise ValueError(f'An invalid Upsample was selected. Please choose one of {SampleMods}')
        super().__init__(*m)
```

::analysis_notes::
logic_invariants:
  - keep shapes consistent across residual branches
  - normalize attention/conv outputs to avoid drift
performance_notes:
  - use channel-last fast paths only if memory format is consistent
  - reparameterize multi-branch convs before deployment
failure_modes:
  - watch for unstable offsets in dynamic sampling
  - window padding/cropping mistakes cause seams
::end::

## 4. Best-Practice Recipes (LLM-Executable Patterns)

### 4.1 Core Architecture Recipe (Hybrid Restoration + SR)
::recipe::
name: hybrid_restoration_sr
inputs: [x: Bx3xHxW, scale: int]
outputs: [y: Bx3x(H*scale)x(W*scale)]
steps:
  - stem = Conv3x3(x)
  - (low, high) = FreqSplit(stem)
  - low = EncoderLow(low)
  - high = EncoderHigh(high)
  - bottleneck = CrossFuse(low, high)
  - dec = Decoder(bottleneck, skips)
  - detail = MoERefine(dec)
  - out = Upsample(detail, scale)
constraints:
  - attention only at mid/low resolution
  - FFT blocks only on low-resolution features
  - moesr-style MSG allowed inside encoder/decoder
::end::

### 4.2 PyTorch Implementation Skeleton (commented, LLM-targeted)
```python
class HybridRestorationSR(nn.Module):
    def __init__(self, in_ch=3, base=48, scale=2, win=8, experts=4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.low_path = nn.Sequential(
            GatedCNNBlock(base),
            GatedCNNBlock(base),
        )
        self.high_path = nn.Sequential(
            WindowAttentionBlock(base, win),
            TokenMoE(base, experts),
        )
        self.fuse = nn.Conv2d(base * 2, base, 1)
        self.reparam_head = ReparamLargeKernel(base)
        self.upsample = UniUpsample("pixelshuffle", scale, base, 3, base)

    def forward(self, x):
        base = self.stem(x)
        low, high = freq_split(base)
        low = self.low_path(low)
        high = self.high_path(high)
        fused = self.fuse(torch.cat([low, high], dim=1))
        fused = self.reparam_head(fused)
        return self.upsample(fused)
```

### 4.3 Design-by-Invariants Checklist
::checklist::
invariant_checks:
  - verify window padding and unpadding correctness
  - verify residual addition uses same dtype and device
  - verify MoE only acts on high-frequency branch
  - verify reparameterized conv outputs match training branch within tolerance
  - verify FFT path preserves mean intensity (no global bias drift)
  - verify upsampler output size equals scale*input size
::end::

## 5. Large-Scale LLM Hints (token-based, deterministic)

### 5.1 Module Library (LLM-tokenized blocks)
::module_library::
- name: freq_split
  core: low=blur(x), high=x-up(low)
  notes: use AvgPool or depthwise blur, keep channel alignment
- name: cross_fusion
  core: concat+1x1 or cross-attn
  notes: use gate or channel-attention to avoid overmixing
- name: fourier_unit
  core: rfft2 -> conv -> irfft2
  notes: normalize and gate to avoid ringing
- name: reparam_conv
  core: train multi-branch, infer single
  notes: verify with numerical equivalence test
- name: gated_cnn
  core: conv1 -> split -> depthwise mix -> conv2
  notes: gamma scaling for stability
- name: window_attention
  core: local attention windows
  notes: shift windows to reduce seams
- name: focused_linear_attention
  core: downsampled long-range
  notes: keep head_dim small for efficiency
- name: token_moe
  core: top-k expert routing
  notes: use load-balancing loss
- name: dysample
  core: predict offsets -> grid_sample
  notes: initialize scope=0 for stability
- name: pixelshuffle
  core: conv->pixelshuffle
  notes: avoid checkerboard via pre-conv
::end::

### 5.2 Failure Modes -> Mitigations (LLM rules)
::failure_mitigations::
- name: moe_collapse
  mitigation: add router z-loss, increase entropy
  monitor: monitor expert utilization histogram
- name: window_seams
  mitigation: use overlap or shift + mask
  monitor: pad to window size and crop
- name: fft_ringing
  mitigation: add frequency loss and gate weights
  monitor: cap FFT gain with sigmoid
- name: dysample_instability
  mitigation: scope init 0, clamp offsets
  monitor: avoid FP16 grid_sample when unstable
- name: kernel_overfit
  mitigation: regularize kernel embedding
  monitor: limit kernel size or augment kernels
- name: oversmoothing
  mitigation: ensure high-frequency path has enough capacity
  monitor: use residual mixing and skip
::end::

## 6. Performance & Complexity Guidance

### 6.1 Complexity Estimation (rule-of-thumb)
- large_kernel_conv: O(K^2 * H * W * C)
- window_attention: O(N_windows * ws^2 * C * heads)
- fft_unit: O(H * W * log(HW) * C)
- moe: O(K_selected * H * W * C)

### 6.2 Where to spend compute
- high-frequency branch: prefer local attention + MoE for textures
- low-frequency branch: prefer gated CNN for stability
- bottleneck: allow lightweight global mixer (SSM/linear attention)

### 6.3 Memory guardrails
- avoid global MHSA at full resolution
- use activation checkpointing only on attention stages
- keep FFT blocks at low resolution to reduce memory

## 7. Training Protocol (LLM-executable)
::training::
losses:
  - l1_reconstruction
  - charbonnier
  - perceptual_vgg
  - frequency_l1
  - optional_gan
schedule:
  - warmup_recon_only: 10-20k iters
  - enable_perceptual: mid-training
  - enable_gan: final 10-20%
moe_constraints:
  - add_load_balance_loss
  - log_expert_usage
  - gradual_topk_ramp
::end::

## 8. Validation & Debugging
::debug_checklist::
sanity_checks:
  - forward shape matches expected scale
  - gradients non-zero through MoE router
  - reparam equivalence (max abs diff < 1e-4)
  - window attention mask aligns with shift
  - FFT output mean close to input mean
profiling:
  - log runtime per block on 256^2 and 512^2 inputs
  - monitor VRAM at each stage
::end::

## 9. Extended LLM Hints and Structured Rules

### 9.1 LLM Rule Pack 1
::llm_rules::
rule_set:
  - id: R01.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R01.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R01.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R01.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R01.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R01.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R01.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R01.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R01.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R01.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R01.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R01.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R01.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R01.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R01.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R01.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R01.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R01.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R01.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R01.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R01.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R01.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R01.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R01.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R01.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R01.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R01.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R01.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R01.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R01.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R01.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R01.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R01.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R01.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R01.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R01.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R01.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R01.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R01.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R01.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.2 LLM Rule Pack 2
::llm_rules::
rule_set:
  - id: R02.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R02.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R02.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R02.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R02.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R02.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R02.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R02.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R02.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R02.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R02.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R02.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R02.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R02.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R02.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R02.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R02.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R02.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R02.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R02.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R02.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R02.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R02.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R02.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R02.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R02.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R02.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R02.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R02.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R02.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R02.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R02.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R02.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R02.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R02.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R02.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R02.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R02.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R02.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R02.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.3 LLM Rule Pack 3
::llm_rules::
rule_set:
  - id: R03.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R03.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R03.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R03.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R03.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R03.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R03.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R03.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R03.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R03.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R03.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R03.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R03.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R03.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R03.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R03.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R03.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R03.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R03.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R03.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R03.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R03.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R03.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R03.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R03.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R03.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R03.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R03.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R03.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R03.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R03.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R03.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R03.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R03.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R03.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R03.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R03.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R03.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R03.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R03.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.4 LLM Rule Pack 4
::llm_rules::
rule_set:
  - id: R04.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R04.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R04.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R04.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R04.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R04.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R04.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R04.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R04.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R04.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R04.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R04.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R04.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R04.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R04.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R04.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R04.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R04.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R04.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R04.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R04.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R04.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R04.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R04.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R04.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R04.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R04.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R04.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R04.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R04.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R04.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R04.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R04.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R04.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R04.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R04.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R04.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R04.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R04.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R04.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.5 LLM Rule Pack 5
::llm_rules::
rule_set:
  - id: R05.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R05.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R05.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R05.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R05.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R05.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R05.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R05.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R05.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R05.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R05.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R05.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R05.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R05.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R05.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R05.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R05.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R05.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R05.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R05.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R05.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R05.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R05.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R05.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R05.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R05.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R05.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R05.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R05.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R05.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R05.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R05.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R05.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R05.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R05.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R05.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R05.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R05.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R05.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R05.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.6 LLM Rule Pack 6
::llm_rules::
rule_set:
  - id: R06.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R06.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R06.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R06.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R06.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R06.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R06.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R06.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R06.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R06.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R06.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R06.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R06.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R06.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R06.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R06.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R06.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R06.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R06.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R06.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R06.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R06.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R06.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R06.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R06.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R06.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R06.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R06.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R06.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R06.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R06.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R06.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R06.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R06.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R06.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R06.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R06.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R06.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R06.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R06.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.7 LLM Rule Pack 7
::llm_rules::
rule_set:
  - id: R07.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R07.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R07.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R07.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R07.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R07.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R07.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R07.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R07.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R07.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R07.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R07.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R07.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R07.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R07.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R07.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R07.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R07.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R07.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R07.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R07.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R07.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R07.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R07.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R07.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R07.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R07.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R07.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R07.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R07.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R07.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R07.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R07.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R07.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R07.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R07.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R07.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R07.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R07.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R07.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.8 LLM Rule Pack 8
::llm_rules::
rule_set:
  - id: R08.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R08.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R08.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R08.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R08.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R08.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R08.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R08.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R08.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R08.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R08.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R08.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R08.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R08.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R08.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R08.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R08.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R08.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R08.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R08.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R08.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R08.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R08.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R08.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R08.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R08.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R08.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R08.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R08.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R08.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R08.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R08.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R08.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R08.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R08.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R08.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R08.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R08.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R08.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R08.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.9 LLM Rule Pack 9
::llm_rules::
rule_set:
  - id: R09.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R09.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R09.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R09.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R09.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R09.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R09.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R09.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R09.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R09.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R09.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R09.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R09.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R09.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R09.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R09.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R09.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R09.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R09.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R09.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R09.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R09.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R09.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R09.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R09.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R09.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R09.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R09.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R09.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R09.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R09.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R09.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R09.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R09.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R09.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R09.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R09.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R09.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R09.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R09.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.10 LLM Rule Pack 10
::llm_rules::
rule_set:
  - id: R10.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R10.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R10.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R10.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R10.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R10.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R10.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R10.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R10.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R10.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R10.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R10.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R10.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R10.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R10.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R10.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R10.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R10.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R10.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R10.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R10.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R10.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R10.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R10.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R10.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R10.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R10.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R10.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R10.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R10.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R10.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R10.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R10.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R10.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R10.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R10.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R10.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R10.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R10.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R10.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.11 LLM Rule Pack 11
::llm_rules::
rule_set:
  - id: R11.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R11.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R11.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R11.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R11.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R11.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R11.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R11.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R11.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R11.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R11.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R11.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R11.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R11.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R11.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R11.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R11.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R11.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R11.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R11.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R11.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R11.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R11.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R11.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R11.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R11.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R11.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R11.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R11.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R11.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R11.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R11.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R11.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R11.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R11.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R11.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R11.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R11.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R11.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R11.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.12 LLM Rule Pack 12
::llm_rules::
rule_set:
  - id: R12.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R12.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R12.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R12.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R12.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R12.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R12.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R12.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R12.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R12.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R12.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R12.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R12.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R12.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R12.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R12.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R12.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R12.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R12.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R12.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R12.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R12.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R12.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R12.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R12.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R12.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R12.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R12.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R12.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R12.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R12.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R12.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R12.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R12.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R12.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R12.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R12.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R12.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R12.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R12.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.13 LLM Rule Pack 13
::llm_rules::
rule_set:
  - id: R13.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R13.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R13.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R13.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R13.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R13.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R13.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R13.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R13.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R13.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R13.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R13.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R13.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R13.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R13.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R13.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R13.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R13.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R13.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R13.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R13.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R13.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R13.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R13.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R13.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R13.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R13.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R13.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R13.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R13.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R13.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R13.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R13.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R13.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R13.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R13.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R13.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R13.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R13.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R13.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.14 LLM Rule Pack 14
::llm_rules::
rule_set:
  - id: R14.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R14.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R14.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R14.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R14.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R14.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R14.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R14.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R14.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R14.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R14.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R14.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R14.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R14.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R14.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R14.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R14.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R14.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R14.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R14.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R14.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R14.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R14.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R14.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R14.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R14.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R14.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R14.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R14.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R14.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R14.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R14.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R14.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R14.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R14.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R14.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R14.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R14.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R14.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R14.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.15 LLM Rule Pack 15
::llm_rules::
rule_set:
  - id: R15.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R15.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R15.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R15.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R15.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R15.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R15.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R15.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R15.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R15.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R15.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R15.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R15.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R15.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R15.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R15.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R15.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R15.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R15.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R15.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R15.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R15.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R15.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R15.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R15.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R15.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R15.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R15.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R15.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R15.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R15.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R15.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R15.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R15.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R15.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R15.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R15.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R15.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R15.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R15.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.16 LLM Rule Pack 16
::llm_rules::
rule_set:
  - id: R16.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R16.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R16.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R16.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R16.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R16.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R16.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R16.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R16.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R16.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R16.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R16.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R16.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R16.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R16.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R16.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R16.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R16.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R16.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R16.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R16.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R16.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R16.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R16.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R16.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R16.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R16.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R16.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R16.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R16.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R16.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R16.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R16.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R16.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R16.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R16.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R16.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R16.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R16.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R16.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.17 LLM Rule Pack 17
::llm_rules::
rule_set:
  - id: R17.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R17.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R17.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R17.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R17.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R17.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R17.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R17.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R17.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R17.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R17.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R17.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R17.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R17.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R17.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R17.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R17.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R17.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R17.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R17.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R17.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R17.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R17.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R17.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R17.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R17.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R17.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R17.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R17.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R17.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R17.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R17.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R17.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R17.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R17.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R17.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R17.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R17.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R17.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R17.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.18 LLM Rule Pack 18
::llm_rules::
rule_set:
  - id: R18.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R18.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R18.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R18.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R18.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R18.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R18.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R18.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R18.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R18.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R18.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R18.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R18.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R18.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R18.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R18.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R18.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R18.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R18.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R18.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R18.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R18.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R18.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R18.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R18.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R18.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R18.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R18.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R18.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R18.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R18.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R18.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R18.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R18.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R18.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R18.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R18.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R18.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R18.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R18.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.19 LLM Rule Pack 19
::llm_rules::
rule_set:
  - id: R19.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R19.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R19.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R19.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R19.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R19.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R19.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R19.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R19.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R19.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R19.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R19.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R19.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R19.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R19.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R19.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R19.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R19.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R19.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R19.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R19.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R19.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R19.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R19.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R19.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R19.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R19.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R19.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R19.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R19.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R19.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R19.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R19.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R19.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R19.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R19.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R19.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R19.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R19.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R19.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.20 LLM Rule Pack 20
::llm_rules::
rule_set:
  - id: R20.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R20.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R20.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R20.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R20.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R20.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R20.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R20.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R20.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R20.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R20.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R20.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R20.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R20.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R20.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R20.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R20.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R20.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R20.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R20.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R20.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R20.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R20.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R20.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R20.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R20.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R20.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R20.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R20.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R20.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R20.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R20.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R20.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R20.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R20.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R20.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R20.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R20.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R20.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R20.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.21 LLM Rule Pack 21
::llm_rules::
rule_set:
  - id: R21.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R21.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R21.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R21.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R21.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R21.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R21.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R21.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R21.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R21.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R21.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R21.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R21.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R21.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R21.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R21.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R21.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R21.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R21.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R21.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R21.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R21.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R21.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R21.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R21.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R21.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R21.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R21.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R21.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R21.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R21.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R21.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R21.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R21.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R21.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R21.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R21.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R21.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R21.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R21.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.22 LLM Rule Pack 22
::llm_rules::
rule_set:
  - id: R22.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R22.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R22.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R22.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R22.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R22.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R22.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R22.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R22.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R22.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R22.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R22.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R22.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R22.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R22.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R22.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R22.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R22.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R22.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R22.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R22.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R22.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R22.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R22.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R22.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R22.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R22.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R22.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R22.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R22.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R22.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R22.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R22.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R22.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R22.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R22.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R22.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R22.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R22.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R22.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.23 LLM Rule Pack 23
::llm_rules::
rule_set:
  - id: R23.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R23.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R23.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R23.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R23.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R23.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R23.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R23.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R23.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R23.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R23.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R23.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R23.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R23.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R23.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R23.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R23.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R23.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R23.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R23.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R23.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R23.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R23.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R23.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R23.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R23.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R23.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R23.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R23.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R23.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R23.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R23.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R23.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R23.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R23.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R23.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R23.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R23.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R23.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R23.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.24 LLM Rule Pack 24
::llm_rules::
rule_set:
  - id: R24.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R24.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R24.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R24.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R24.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R24.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R24.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R24.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R24.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R24.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R24.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R24.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R24.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R24.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R24.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R24.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R24.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R24.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R24.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R24.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R24.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R24.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R24.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R24.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R24.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R24.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R24.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R24.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R24.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R24.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R24.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R24.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R24.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R24.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R24.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R24.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R24.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R24.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R24.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R24.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.25 LLM Rule Pack 25
::llm_rules::
rule_set:
  - id: R25.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R25.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R25.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R25.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R25.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R25.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R25.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R25.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R25.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R25.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R25.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R25.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R25.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R25.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R25.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R25.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R25.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R25.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R25.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R25.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R25.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R25.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R25.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R25.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R25.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R25.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R25.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R25.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R25.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R25.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R25.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R25.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R25.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R25.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R25.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R25.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R25.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R25.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R25.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R25.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.26 LLM Rule Pack 26
::llm_rules::
rule_set:
  - id: R26.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R26.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R26.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R26.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R26.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R26.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R26.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R26.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R26.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R26.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R26.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R26.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R26.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R26.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R26.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R26.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R26.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R26.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R26.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R26.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R26.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R26.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R26.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R26.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R26.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R26.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R26.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R26.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R26.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R26.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R26.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R26.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R26.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R26.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R26.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R26.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R26.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R26.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R26.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R26.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.27 LLM Rule Pack 27
::llm_rules::
rule_set:
  - id: R27.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R27.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R27.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R27.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R27.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R27.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R27.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R27.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R27.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R27.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R27.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R27.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R27.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R27.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R27.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R27.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R27.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R27.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R27.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R27.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R27.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R27.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R27.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R27.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R27.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R27.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R27.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R27.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R27.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R27.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R27.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R27.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R27.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R27.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R27.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R27.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R27.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R27.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R27.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R27.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.28 LLM Rule Pack 28
::llm_rules::
rule_set:
  - id: R28.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R28.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R28.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R28.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R28.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R28.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R28.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R28.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R28.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R28.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R28.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R28.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R28.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R28.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R28.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R28.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R28.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R28.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R28.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R28.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R28.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R28.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R28.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R28.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R28.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R28.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R28.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R28.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R28.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R28.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R28.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R28.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R28.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R28.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R28.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R28.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R28.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R28.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R28.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R28.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.29 LLM Rule Pack 29
::llm_rules::
rule_set:
  - id: R29.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R29.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R29.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R29.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R29.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R29.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R29.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R29.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R29.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R29.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R29.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R29.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R29.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R29.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R29.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R29.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R29.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R29.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R29.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R29.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R29.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R29.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R29.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R29.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R29.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R29.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R29.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R29.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R29.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R29.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R29.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R29.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R29.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R29.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R29.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R29.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R29.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R29.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R29.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R29.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.30 LLM Rule Pack 30
::llm_rules::
rule_set:
  - id: R30.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R30.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R30.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R30.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R30.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R30.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R30.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R30.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R30.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R30.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R30.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R30.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R30.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R30.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R30.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R30.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R30.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R30.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R30.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R30.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R30.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R30.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R30.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R30.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R30.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R30.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R30.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R30.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R30.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R30.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R30.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R30.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R30.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R30.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R30.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R30.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R30.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R30.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R30.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R30.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.31 LLM Rule Pack 31
::llm_rules::
rule_set:
  - id: R31.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R31.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R31.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R31.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R31.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R31.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R31.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R31.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R31.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R31.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R31.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R31.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R31.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R31.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R31.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R31.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R31.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R31.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R31.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R31.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R31.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R31.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R31.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R31.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R31.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R31.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R31.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R31.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R31.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R31.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R31.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R31.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R31.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R31.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R31.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R31.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R31.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R31.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R31.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R31.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.32 LLM Rule Pack 32
::llm_rules::
rule_set:
  - id: R32.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R32.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R32.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R32.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R32.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R32.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R32.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R32.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R32.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R32.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R32.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R32.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R32.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R32.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R32.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R32.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R32.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R32.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R32.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R32.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R32.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R32.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R32.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R32.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R32.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R32.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R32.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R32.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R32.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R32.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R32.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R32.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R32.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R32.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R32.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R32.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R32.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R32.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R32.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R32.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.33 LLM Rule Pack 33
::llm_rules::
rule_set:
  - id: R33.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R33.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R33.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R33.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R33.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R33.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R33.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R33.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R33.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R33.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R33.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R33.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R33.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R33.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R33.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R33.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R33.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R33.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R33.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R33.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R33.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R33.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R33.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R33.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R33.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R33.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R33.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R33.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R33.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R33.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R33.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R33.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R33.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R33.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R33.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R33.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R33.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R33.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R33.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R33.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.34 LLM Rule Pack 34
::llm_rules::
rule_set:
  - id: R34.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R34.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R34.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R34.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R34.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R34.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R34.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R34.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R34.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R34.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R34.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R34.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R34.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R34.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R34.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R34.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R34.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R34.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R34.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R34.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R34.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R34.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R34.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R34.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R34.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R34.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R34.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R34.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R34.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R34.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R34.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R34.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R34.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R34.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R34.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R34.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R34.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R34.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R34.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R34.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.35 LLM Rule Pack 35
::llm_rules::
rule_set:
  - id: R35.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R35.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R35.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R35.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R35.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R35.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R35.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R35.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R35.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R35.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R35.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R35.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R35.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R35.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R35.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R35.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R35.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R35.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R35.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R35.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R35.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R35.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R35.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R35.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R35.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R35.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R35.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R35.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R35.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R35.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R35.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R35.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R35.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R35.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R35.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R35.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R35.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R35.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R35.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R35.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.36 LLM Rule Pack 36
::llm_rules::
rule_set:
  - id: R36.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R36.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R36.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R36.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R36.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R36.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R36.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R36.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R36.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R36.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R36.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R36.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R36.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R36.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R36.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R36.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R36.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R36.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R36.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R36.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R36.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R36.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R36.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R36.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R36.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R36.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R36.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R36.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R36.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R36.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R36.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R36.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R36.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R36.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R36.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R36.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R36.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R36.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R36.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R36.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.37 LLM Rule Pack 37
::llm_rules::
rule_set:
  - id: R37.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R37.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R37.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R37.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R37.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R37.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R37.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R37.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R37.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R37.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R37.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R37.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R37.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R37.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R37.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R37.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R37.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R37.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R37.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R37.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R37.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R37.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R37.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R37.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R37.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R37.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R37.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R37.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R37.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R37.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R37.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R37.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R37.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R37.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R37.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R37.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R37.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R37.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R37.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R37.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.38 LLM Rule Pack 38
::llm_rules::
rule_set:
  - id: R38.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R38.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R38.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R38.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R38.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R38.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R38.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R38.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R38.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R38.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R38.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R38.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R38.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R38.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R38.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R38.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R38.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R38.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R38.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R38.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R38.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R38.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R38.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R38.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R38.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R38.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R38.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R38.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R38.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R38.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R38.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R38.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R38.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R38.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R38.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R38.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R38.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R38.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R38.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R38.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.39 LLM Rule Pack 39
::llm_rules::
rule_set:
  - id: R39.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R39.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R39.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R39.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R39.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R39.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R39.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R39.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R39.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R39.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R39.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R39.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R39.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R39.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R39.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R39.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R39.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R39.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R39.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R39.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R39.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R39.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R39.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R39.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R39.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R39.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R39.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R39.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R39.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R39.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R39.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R39.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R39.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R39.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R39.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R39.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R39.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R39.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R39.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R39.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

### 9.40 LLM Rule Pack 40
::llm_rules::
rule_set:
  - id: R40.01
    intent: enforce_shape_consistency_1
    action: assert_tensor_dims_and_channels_1
    failure_signal: mismatch_1
  - id: R40.02
    intent: enforce_shape_consistency_2
    action: assert_tensor_dims_and_channels_2
    failure_signal: mismatch_2
  - id: R40.03
    intent: enforce_shape_consistency_3
    action: assert_tensor_dims_and_channels_3
    failure_signal: mismatch_3
  - id: R40.04
    intent: enforce_shape_consistency_4
    action: assert_tensor_dims_and_channels_4
    failure_signal: mismatch_4
  - id: R40.05
    intent: enforce_shape_consistency_5
    action: assert_tensor_dims_and_channels_5
    failure_signal: mismatch_5
  - id: R40.06
    intent: enforce_shape_consistency_6
    action: assert_tensor_dims_and_channels_6
    failure_signal: mismatch_6
  - id: R40.07
    intent: enforce_shape_consistency_7
    action: assert_tensor_dims_and_channels_7
    failure_signal: mismatch_7
  - id: R40.08
    intent: enforce_shape_consistency_8
    action: assert_tensor_dims_and_channels_8
    failure_signal: mismatch_8
  - id: R40.09
    intent: enforce_shape_consistency_9
    action: assert_tensor_dims_and_channels_9
    failure_signal: mismatch_9
  - id: R40.10
    intent: enforce_shape_consistency_10
    action: assert_tensor_dims_and_channels_10
    failure_signal: mismatch_10
  - id: R40.11
    intent: enforce_shape_consistency_11
    action: assert_tensor_dims_and_channels_11
    failure_signal: mismatch_11
  - id: R40.12
    intent: enforce_shape_consistency_12
    action: assert_tensor_dims_and_channels_12
    failure_signal: mismatch_12
  - id: R40.13
    intent: enforce_shape_consistency_13
    action: assert_tensor_dims_and_channels_13
    failure_signal: mismatch_13
  - id: R40.14
    intent: enforce_shape_consistency_14
    action: assert_tensor_dims_and_channels_14
    failure_signal: mismatch_14
  - id: R40.15
    intent: enforce_shape_consistency_15
    action: assert_tensor_dims_and_channels_15
    failure_signal: mismatch_15
  - id: R40.16
    intent: enforce_shape_consistency_16
    action: assert_tensor_dims_and_channels_16
    failure_signal: mismatch_16
  - id: R40.17
    intent: enforce_shape_consistency_17
    action: assert_tensor_dims_and_channels_17
    failure_signal: mismatch_17
  - id: R40.18
    intent: enforce_shape_consistency_18
    action: assert_tensor_dims_and_channels_18
    failure_signal: mismatch_18
  - id: R40.19
    intent: enforce_shape_consistency_19
    action: assert_tensor_dims_and_channels_19
    failure_signal: mismatch_19
  - id: R40.20
    intent: enforce_shape_consistency_20
    action: assert_tensor_dims_and_channels_20
    failure_signal: mismatch_20
  - id: R40.21
    intent: enforce_shape_consistency_21
    action: assert_tensor_dims_and_channels_21
    failure_signal: mismatch_21
  - id: R40.22
    intent: enforce_shape_consistency_22
    action: assert_tensor_dims_and_channels_22
    failure_signal: mismatch_22
  - id: R40.23
    intent: enforce_shape_consistency_23
    action: assert_tensor_dims_and_channels_23
    failure_signal: mismatch_23
  - id: R40.24
    intent: enforce_shape_consistency_24
    action: assert_tensor_dims_and_channels_24
    failure_signal: mismatch_24
  - id: R40.25
    intent: enforce_shape_consistency_25
    action: assert_tensor_dims_and_channels_25
    failure_signal: mismatch_25
  - id: R40.26
    intent: enforce_shape_consistency_26
    action: assert_tensor_dims_and_channels_26
    failure_signal: mismatch_26
  - id: R40.27
    intent: enforce_shape_consistency_27
    action: assert_tensor_dims_and_channels_27
    failure_signal: mismatch_27
  - id: R40.28
    intent: enforce_shape_consistency_28
    action: assert_tensor_dims_and_channels_28
    failure_signal: mismatch_28
  - id: R40.29
    intent: enforce_shape_consistency_29
    action: assert_tensor_dims_and_channels_29
    failure_signal: mismatch_29
  - id: R40.30
    intent: enforce_shape_consistency_30
    action: assert_tensor_dims_and_channels_30
    failure_signal: mismatch_30
  - id: R40.31
    intent: enforce_shape_consistency_31
    action: assert_tensor_dims_and_channels_31
    failure_signal: mismatch_31
  - id: R40.32
    intent: enforce_shape_consistency_32
    action: assert_tensor_dims_and_channels_32
    failure_signal: mismatch_32
  - id: R40.33
    intent: enforce_shape_consistency_33
    action: assert_tensor_dims_and_channels_33
    failure_signal: mismatch_33
  - id: R40.34
    intent: enforce_shape_consistency_34
    action: assert_tensor_dims_and_channels_34
    failure_signal: mismatch_34
  - id: R40.35
    intent: enforce_shape_consistency_35
    action: assert_tensor_dims_and_channels_35
    failure_signal: mismatch_35
  - id: R40.36
    intent: enforce_shape_consistency_36
    action: assert_tensor_dims_and_channels_36
    failure_signal: mismatch_36
  - id: R40.37
    intent: enforce_shape_consistency_37
    action: assert_tensor_dims_and_channels_37
    failure_signal: mismatch_37
  - id: R40.38
    intent: enforce_shape_consistency_38
    action: assert_tensor_dims_and_channels_38
    failure_signal: mismatch_38
  - id: R40.39
    intent: enforce_shape_consistency_39
    action: assert_tensor_dims_and_channels_39
    failure_signal: mismatch_39
  - id: R40.40
    intent: enforce_shape_consistency_40
    action: assert_tensor_dims_and_channels_40
    failure_signal: mismatch_40
::end::

