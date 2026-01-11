import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import trunc_normal_
from einops import rearrange
from .arch_util import make_layer
from .vision_lstm2 import ViLBlockPair

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = nn.Parameter(res_scale * torch.ones(1), requires_grad=True)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

    def forward(self, x):
        identity = x
        out = self.conv2(self.prelu(self.conv1(x)))
        return identity + out * self.res_scale

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc = nn.Linear(in_features, in_features, bias=bias)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x, x_size):

        x_fc = torch.sigmoid(self.fc(x))

        x1 = self.fc1(x)
        x1 = torch.square(torch.relu(x1))
        x = self.fc2(x1) * x_fc

        return x

class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, drop_path: float = 0,
                 key_norm=False):
        super().__init__()
        
        hidden_features = int(n_embd * hidden_rate)

        self.ln = nn.LayerNorm(n_embd) 

        self.att = ViLBlockPair(n_embd, conv_kind="causal1d", conv_kernel_size=4)
        self.ffn = Mlp(in_features=n_embd, hidden_features=hidden_features, bias=False)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x): 
        b, c, h, w = x.shape
        
        resolution = (h, w)

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma1 * self.att(x, resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        x = rearrange(x, 'b c h w -> b (h w) c')    
        x = x + self.gamma2 * self.ffn(self.ln(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

class BlockWCNNs(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., CNN_num=12):
        super().__init__()
        self.lstm = Block(n_embd=dim, hidden_rate=mlp_ratio, drop_path=drop_path)
        self.cnn = make_layer(ResidualBlockNoBN, CNN_num, num_feat=dim, res_scale=1)
        self.conv = nn.Conv2d(dim * 2, dim, 3, 1, 1)

    def forward(self, x):
        x_lstm = self.lstm(x)
        x = self.cnn(x)
        x = self.conv(torch.cat((x, x_lstm), dim=1))
        return x

class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 CNN_num=12,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(BlockWCNNs(dim=dim, mlp_ratio=mlp_ratio, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, CNN_num = CNN_num,))
            
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class ResidualGroup(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 CNN_num=12,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 resi_connection='1conv'):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        # self.input_resolution = [48, 48] # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            CNN_num=CNN_num,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x):
        return self.conv(self.residual_group(x)) + x

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

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
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@ARCH_REGISTRY.register()
class LSTMConvSR(nn.Module):
    def __init__(self,
                 upscale=2,
                 img_range=1.,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 mlp_ratio=4.,
                 CNN_num=12,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 **kwargs):
        super(LSTMConvSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:            
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        self.CNN_num = CNN_num
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Neural Block (NB)
        self.NB_layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4-layer
            layer = ResidualGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                CNN_num=self.CNN_num,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
            )
            self.NB_layers.append(layer)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            x = self.NB_layers[i](x)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)

            x = self.conv_after_body(self.forward_features(x)) + x

            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x