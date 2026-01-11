import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, gap=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            AvgMax(in_channel) if gap else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
    

class AvgBranch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):
        
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out
    
class LocalAvgBranch(nn.Module):
    def __init__(self, dim) -> None:
        super(LocalAvgBranch, self).__init__()

        self.gap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):
        
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out

class StripGlobalAvgBranch(nn.Module):
    def __init__(self, dim, size) -> None:
        super(StripGlobalAvgBranch, self).__init__()

        
        self.gap = nn.AdaptiveAvgPool2d(size)

        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):
        
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out



class StripGlobalMaxBranch(nn.Module):
    def __init__(self, dim, kernel) -> None:
        super().__init__()

        self.mp = nn.AdaptiveMaxPool2d((kernel))
        
        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):

        high_frequency = self.mp(x)
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class MaxBranch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):

        high_frequency = self.mp(x)
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out
    

class MaxDilateBranch(nn.Module):
    def __init__(self, dim) -> None:
        super(MaxDilateBranch, self).__init__()

        dilation = 2
        kernel = 5

        self.pad = nn.ReflectionPad2d(dilation*(kernel-1)//2)

        self.mp = nn.MaxPool2d(kernel_size=kernel, dilation=dilation, stride=1)
        
        self.low_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim,1,1), requires_grad=True)

    def forward(self, x):

        high_frequency = self.mp(self.pad(x))
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class AvgMax(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gap = AvgBranch(dim)
        self.mp = MaxBranch(dim)
        self.local_gap = LocalAvgBranch(dim)
        self.globa_horizontal_avg = StripGlobalAvgBranch(dim, (None,1))
        self.globa_vertical_avg = StripGlobalAvgBranch(dim, (1,None))
        self.global_horizontal_max = StripGlobalMaxBranch(dim, (None,1))
        self.global_vertial_max = StripGlobalMaxBranch(dim, (1, None))
        self.maxdilation = MaxDilateBranch(dim)
    def forward(self, x):
        x1 = self.gap(x)
        x2 = self.mp(x)
        x3 = self.local_gap(x)
        x4 = self.globa_horizontal_avg(x)
        x5 = self.globa_vertical_avg(x)
        x6 = self.global_horizontal_max(x)
        x7 = self.global_vertial_max(x)
        x8 = self.maxdilation(x)
        return x1+x2+x3+x4+x5+x6+x7+x8



