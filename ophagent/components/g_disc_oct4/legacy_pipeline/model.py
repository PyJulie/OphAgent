import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import *
# from torchinfo import summary
import random

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms = True
# torch.cuda.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)


class SpatialTransformer2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # 自分で変更
        # size = (224, 48, 11)
        # size = (28, 6, 11)

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)

        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        # We only focus on shift in one dimension, so the other two shift is zero.
        oshift1 = torch.zeros_like(flow)
        oshift2 = torch.zeros_like(flow)
        flow = torch.cat([flow, oshift1, oshift2], 1)

        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def computeMuVariance(x):
    '''
    Compute the mean and variance along H direction of each surface.
    '''
    device = x.device

    B, N, H, W = x.size()  # Num is the num of surface for each patientD

    # compute mu
    Y = torch.arange(H).view((1, 1, H, 1)).expand(
        x.size()).to(device=device, dtype=torch.int16)

    # use slice method to compute P*Y
    for b in range(B):
        if 0 == b:
            PY = (x[b, ]*Y[b, ]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (x[b, ]*Y[b, ]).unsqueeze(dim=0)))

    mu = torch.sum(PY, dim=-2, keepdim=True)  # size: B,N,1,W
    del PY  # hope to free memory.
    # Mu = mu.expand(x.size())

    # this slice method is to avoid using big GPU memory .
    # for b in range(B):
    #     if 0 == b:
    #         sigma2 = torch.sum(x[b, ]*torch.pow(Y[b, ]-Mu[b, ], 2),
    #                            dim=-3, keepdim=False).unsqueeze(dim=0)
    #     else:
    #         sigma2 = torch.cat((sigma2, torch.sum(
    #             x[b, ]*torch.pow(Y[b, ]-Mu[b, ], 2), dim=-3, keepdim=False).unsqueeze(dim=0)))

    return mu.squeeze(dim=-2)


def logits2Prob(x, dim):
    # convert logits to probability for input x
    xMaxDim, _ = torch.max(x, dim=dim, keepdim=True)
    xMaxDim = xMaxDim.expand_as(x)
    # using inputMaxDim is to avoid overflow.
    prob = F.softmax(x - xMaxDim, dim=dim)
    return prob


def channel_wise_softmax(x):
    dim = 1
    # convert logits to probability for input x
    xMaxDim, _ = torch.max(x, dim=dim, keepdim=True)
    xMaxDim = xMaxDim.expand_as(x)
    # using inputMaxDim is to avoid overflow.
    prob = F.softmax(x - xMaxDim, dim=dim)
    return prob


def column_wise_softmax(x):
    dim = 2
    # convert logits to probability for input x
    xMaxDim, _ = torch.max(x, dim=dim, keepdim=True)
    xMaxDim = xMaxDim.expand_as(x)
    # using inputMaxDim is to avoid overflow.
    prob = F.softmax(x - xMaxDim, dim=dim)
    return prob


# def _softmax(x, beta=10):
#     c = np.max(beta*x)
#     ex = np.exp(beta*x - c)
#     sum_ex = np.sum(ex)
#     return ex / sum_ex


# def softargmax_2D(x):
#     assert x.ndim == 4, "x dim must be 4"

#     s = _softmax(x)
#     x = np.arange(x.shape[1])
#     y = np.arange(x.shape[0])
#     xx, yy = np.meshgrid(x, y)
#     xmax = np.sum(s*xx)
#     ymax = np.sum(s*yy)
#     return ymax, xmax

def softargmax(x, beta=10):
    h = x.shape[2]
    a = torch.exp(beta*x)
    b = torch.sum(a, 2)
    a = a.permute(0, 1, 3, 2)
    b = b.unsqueeze(3)
    _softmax = a/b
    _softmax = _softmax.permute(0, 1, 3, 2)
    pos = torch.tensor(
        range(0, h), device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(3).expand_as(x)
    _softargamax = torch.sum(_softmax * pos, 2)

    return _softargamax


class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 4:
            raise ValueError(
                'expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError(
                'expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        # out = self.activation(self.bn1(self.conv1(x)))
        x = self.conv1(x)
        x = self.bn1(x)
        out = self.activation(x)
        return out


class LUConv2d(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm2d(out_chan)
        # self.se = SELayer2D(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        # out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # out = self.se(out)
        out = self.activation(out)
        return out


def _make_nConv_2d(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv2d(in_channel, depth, act)
        layer2 = LUConv2d(depth, depth, act)
    else:
        layer1 = LUConv2d(in_channel, depth, act)
        layer2 = LUConv2d(depth, depth, act)

    return nn.Sequential(layer1, layer2)


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, depth, act)
        layer2 = LUConv(depth, depth, act)
    else:
        layer1 = LUConv(in_channel, depth, act)
        layer2 = LUConv(depth, depth, act)

    return nn.Sequential(layer1, layer2)


class NormalConv(nn.Module):
    def __init__(self, inChans, outChans, act):
        super(NormalConv, self).__init__()
        self.ops = _make_nConv(inChans, outChans, act, double_chnnel=False)

    def forward(self, x):
        out = self.ops(x)
        return out


class DownTransition2D(nn.Module):
    def __init__(self, in_channel, depth, act, pol=True):
        super(DownTransition2D, self).__init__()
        self.ops = _make_nConv_2d(in_channel, depth, act)
        self.maxpool = nn.MaxPool2d(2)
        # self.current_depth = depth
        self.pol = pol

    def forward(self, x):
        if not self.pol:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


# class UpTransition(nn.Module):
#     def __init__(self, inChans, outChans, depth, act):
#         super(UpTransition, self).__init__()
#         self.depth = depth
#         self.up_conv = nn.ConvTranspose3d(
#             inChans, outChans, kernel_size=(2, 2, 1), stride=(2, 2, 1))
#         self.ops = _make_nConv(outChans*2, outChans, act, double_chnnel=False)

#     def forward(self, x, skip_x):
#         out_up_conv = self.up_conv(x)
#         concat = torch.cat((out_up_conv, skip_x), 1)
#         out = self.ops(concat)
#         return out


# class UpTransition2D(nn.Module):
#     def __init__(self, inChans, outChans, depth, act):
#         super(UpTransition2D, self).__init__()
#         self.depth = depth
#         self.up_conv = nn.ConvTranspose2d(
#             inChans, outChans, kernel_size=(2, 2), stride=(2, 2))
#         self.ops = _make_nConv_2d(
#             outChans*2, outChans, act, double_chnnel=False)

#     def forward(self, x, skip_x):
#         out_up_conv = self.up_conv(x)
#         concat = torch.cat((out_up_conv, skip_x), 1)
#         out = self.ops(concat)
#         return out


# class OutputTransition(nn.Module):
#     def __init__(self, inChans, n_labels):

#         super(OutputTransition, self).__init__()
#         self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out = self.final_conv(x)
#         return out


class OutputTransition2D(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition2D, self).__init__()
        self.final_conv = nn.Conv2d(inChans, n_labels, kernel_size=1)

    def forward(self, x):
        out = self.final_conv(x)
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class ResBlock2D(nn.Module):
    expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        activation='PReLU',
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        if activation == 'PReLU':
            self.prelu1 = nn.PReLU()
            self.prelu2 = nn.PReLU()
        else:
            self.prelu1 = nn.ReLU()
            self.prelu2 = nn.ReLU()
        self.conv3 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        # if in_channels != channels * self.expansion:
        #     self.shortcut = nn.Sequential(
        #         conv1x1(in_channels, channels * self.expansion, stride),
        #         nn.BatchNorm2d(channels * self.expansion),
        #     )
        # else:
        #     self.shortcut = nn.Sequential()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.prelu2(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = conv3x3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        return x


class NormalConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ConvBranch(nn.Module):

    def __init__(
        self,
        in_channel,
        middle_channel,
        out_channnel,
    ):
        super(ConvBranch, self).__init__()
        self.res = ResBlock2D(in_channel, middle_channel)
        self.conv1x1 = conv1x1(middle_channel, out_channnel)

    def forward(self, x):
        out = self.res(x)
        out = self.conv1x1(out)

        return out


class UNet2D(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        super(UNet2D, self).__init__()
        # print('UNet2D_init')

        self.res1 = ResBlock2D(in_channel, 64)
        self.res2 = ResBlock2D(64, 64)
        self.res3 = ResBlock2D(64, 64)
        self.res4 = ResBlock2D(64, 64)
        self.res5 = ResBlock2D(64, 64)
        self.res6 = ResBlock2D(64, 64)

        self.down_res1 = ResBlock2D(128, 64)
        self.down_res2 = ResBlock2D(128, 64)
        self.down_res3 = ResBlock2D(128, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.Upsample(
        #     scale_factor=2, mode='nearest')
        # self.upsample = nn.ConvTranspose2d(
        #     64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv1x1_head = conv1x1(128, 64)

    def encoder(self, x):
        x = self.res1(x)
        skip1 = self.res2(x)
        x = self.maxpool(skip1)

        skip2 = self.res3(x)
        x = self.maxpool(skip2)

        skip3 = self.res4(x)
        x = self.maxpool(skip3)

        skip4 = self.res5(x)
        x = self.maxpool(skip4)

        x = self.res6(x)

        return x, skip1, skip2, skip3, skip4

    def decoder(self, x, skip1, skip2, skip3, skip4):
        x = self.upsample(x)

        x = torch.cat([x, skip4], dim=1)
        x = self.down_res1(x)
        x = self.upsample(x)

        x = torch.cat([x, skip3], dim=1)
        x = self.down_res2(x)
        x = self.upsample(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.down_res3(x)
        x = self.upsample(x)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = torch.cat([x, skip1], dim=1)
        x = self.conv1x1_head(x)

        return x

    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, skip4)
        # x = self.decoder(self.encoder(x))

        return x


class UNet2D_deconv(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        super(UNet2D_deconv, self).__init__()
        print('UNet2D_init')

        self.res1 = ResBlock2D(in_channel, 64)
        self.res2 = ResBlock2D(64, 64)
        self.res3 = ResBlock2D(64, 64)
        self.res4 = ResBlock2D(64, 64)
        self.res5 = ResBlock2D(64, 64)
        self.res6 = ResBlock2D(64, 64)

        self.down_res1 = ResBlock2D(128, 64)
        self.down_res2 = ResBlock2D(128, 64)
        self.down_res3 = ResBlock2D(128, 64)
        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(
        #     scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.Upsample(
        #     scale_factor=2, mode='nearest')
        self.upsample1 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample4 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv1x1_head = conv1x1(128, 64)

    def encoder(self, x):
        x = self.res1(x)
        skip1 = self.res2(x)
        x = self.maxpool(skip1)

        skip2 = self.res3(x)
        x = self.maxpool(skip2)

        skip3 = self.res4(x)
        x = self.maxpool(skip3)

        skip4 = self.res5(x)
        x = self.maxpool(skip4)

        x = self.res6(x)

        return x, skip1, skip2, skip3, skip4

    def decoder(self, x, skip1, skip2, skip3, skip4):
        x = self.upsample1(x)

        x = torch.cat([x, skip4], dim=1)
        x = self.down_res1(x)
        x = self.upsample2(x)

        x = torch.cat([x, skip3], dim=1)
        x = self.down_res2(x)
        x = self.upsample3(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.down_res3(x)
        x = self.upsample4(x)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = torch.cat([x, skip1], dim=1)
        x = self.conv1x1_head(x)

        return x

    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3, skip4)
        # x = self.decoder(self.encoder(x))

        return x


class UNet2D_multiscale(UNet2D):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        UNet2D.__init__(self, n_layer, input_size, in_channel)
        self.conv1x1_1 = conv1x1(64, 64)
        self.conv1x1_2 = conv1x1(64, 64)
        self.conv1x1_3 = conv1x1(64, 64)
        self.conv1x1_4 = conv1x1(64, 64)

    def decoder(self, x, skip1, skip2, skip3, skip4):
        x_list = []
        x_list.append(self.conv1x1_1(x))
        x = self.upsample(x)

        x = torch.cat([x, skip4], dim=1)
        x = self.down_res1(x)
        x_list.append(self.conv1x1_2(x))
        x = self.upsample(x)

        x = torch.cat([x, skip3], dim=1)
        x = self.down_res2(x)
        x_list.append(self.conv1x1_3(x))
        x = self.upsample(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.down_res3(x)
        x_list.append(self.conv1x1_4(x))
        x = self.upsample(x)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = torch.cat([x, skip1], dim=1)
        x = self.conv1x1_head(x)
        x_list.append(x)

        return x, x_list

    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x, x_list = self.decoder(x, skip1, skip2, skip3, skip4)

        return x, x_list


class TwoBranch(nn.Module):
    def __init__(self, n_layer=9, input_size=(128, 1024), use_softargmax=False, channels=(64, 32), has_edema=False):
        super().__init__()
        self.use_softargmax = use_softargmax
        add_layer = 2 if has_edema else 1
        self.conv_1 = ConvBranch(channels[0], channels[1], n_layer+add_layer)
        self.conv_2 = ConvBranch(channels[0], channels[1], n_layer)

    def layer_branch(self, x):
        layer_maps = self.conv_1(x)
        layer_maps = channel_wise_softmax(layer_maps)

        return layer_maps

    def surface_branch(self, x):
        surface_maps = self.conv_2(x)
        surface_maps = column_wise_softmax(surface_maps)
        if self.use_softargmax:
            surface_pos = softargmax(surface_maps)
        else:
            surface_pos = computeMuVariance(surface_maps)
        S = surface_pos.clone()

        return surface_maps, S

    def forward(self, x):
        layer_maps = self.layer_branch(x)
        surface_maps, S = self.surface_branch(x)

        return layer_maps, surface_maps, S


class FCBR(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__()
        self.n_layer = n_layer

        self.unet = UNet2D(n_layer, input_size, in_channel)

        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=True, has_edema=has_edema)

    def topology_module(self, S):
        # relu for Topology Guarantee
        for i in range(1, self.n_layer):
            S[:, i, :] = torch.where(
                S[:, i, :] < S[:, i-1, :], S[:, i-1, :], S[:, i, :])

        return S

    def forward(self, x):
        x = self.unet(x)
        layer_maps, surface_maps, S = self.branch(x)
        S = self.topology_module(S)

        # return S, out, layerProb, mu, x_out, flow
        return {'final_surfaces': S, 'surface_maps': surface_maps, 'layer_maps': layer_maps}


class SLSS(FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__(n_layer, input_size, in_channel, has_edema)
        # self.n_layer = n_layer
        # self.unet = UNet2D(n_layer, input_size, in_channel)

        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, has_edema=has_edema)


class FCBR_new(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__()
        self.n_layer = n_layer

        self.unet = UNet2D(n_layer, input_size, in_channel)

        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=True, has_edema=has_edema)

    def topology_module(self, _S):
        # relu for Topology Guarantee
        S = _S.detach().clone()
        for i in range(1, self.n_layer):
            S[:, i, :] = torch.where(
                S[:, i, :] < S[:, i-1, :], S[:, i-1, :], S[:, i, :])

        return S

    def forward(self, x):
        x = self.unet(x)
        layer_maps, surface_maps, S = self.branch(x)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}



class NewSLSS(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__(n_layer, input_size, in_channel, has_edema)
        self.n_layer = n_layer
        self.unet = UNet2D(n_layer, input_size, in_channel)

        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, has_edema=has_edema)

    def topology_module(self, _S):
        # S = _S.detach().clone()
        S = _S.clone()
        for i in range(1, self.n_layer):
            S[:, i, :] = torch.where(
                S[:, i, :] < S[:, i-1, :], S[:, i-1, :], S[:, i, :])

        return S

    def forward(self, x):
        x = self.unet(x)
        layer_maps, surface_maps, S = self.branch(x)
        guarantee_S = self.topology_module(S)

        # return S, out, layerProb, mu, x_out, flow
        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}



class FCBRwithRefine(FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        FCBR.__init__(
            self, n_layer, input_size, in_channel)
        self.refine_block = NormalConv(2*n_layer+1, 64)
        self.refined_branches = TwoBranch(n_layer)

    def refinement_module(self, layer_maps, surface_maps):
        x = torch.cat([layer_maps, surface_maps], dim=1)
        x = self.refine_block(x)
        return x

    def forward(self, x):
        x = self.unet(x)
        layer_maps, surface_maps, S = self.branch(x)
        out = self.refinement_module(layer_maps, surface_maps)
        _, refined_surface_maps, refined_S = self.refined_branches(out)

        S = self.topology_module(S)
        refined_S = self.topology_module(refined_S)

        # return S, out, layerProb, mu, x_out, flow
        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'refined_final_surface': refined_S,
                'refined_surface_maps': refined_surface_maps}


class SLSSwithRefinePost(FCBR):
    """
    実質SLSS
    トポロジー保証モジュールを後処理にして，
    L1 lossの計算前にトポロジー保証モジュールを入れないようにした．

    Args:
        FCBR (_type_): _description_
    """

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__(n_layer, input_size, in_channel, has_edema)
        add_ch = 2 if has_edema else 1
        self.refine_block = NormalConv(2*n_layer+add_ch, 64)
        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, has_edema=has_edema)
        self.refined_branches = TwoBranch(
            n_layer, input_size, use_softargmax=False, has_edema=has_edema)

    def topology_module(self, _S):
        S = _S.detach().clone()
        for i in range(1, self.n_layer):
            S[:, i, :] = torch.where(
                S[:, i, :] < S[:, i-1, :], S[:, i-1, :], S[:, i, :])

        return S

    def refinement_module(self, layer_maps, surface_maps):
        x = torch.cat([layer_maps, surface_maps], dim=1)
        x = self.refine_block(x)
        return x

    def forward(self, x):
        x = self.unet(x)
        layer_maps, surface_maps, S = self.branch(x)
        out = self.refinement_module(layer_maps, surface_maps)
        _, refined_surface_maps, refined_S = self.refined_branches(out)

        # S = self.topology_module(S)
        guarantee_S = self.topology_module(refined_S)

        # return S, out, layerProb, mu, x_out, flow
        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'refined_final_surface': refined_S,
                'refined_surface_maps': refined_surface_maps, 'guarantee_surface': guarantee_S}


class SLSSRefine(FCBR):
    """
    トポロジー保証モジュールを後処理にして，
    L1 lossの計算前にトポロジー保証モジュールを入れないようにした．

    Args:
        FCBR (_type_): _description_
    """

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__(n_layer, input_size, in_channel, has_edema)
        self.refine_block = NormalConv(2*n_layer+1, 64)
        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, has_edema=has_edema)
        self.refined_branches = TwoBranch(
            n_layer, input_size, use_softargmax=False, has_edema=has_edema)

    def refinement_module(self, layer_maps, surface_maps):
        x = torch.cat([layer_maps, surface_maps], dim=1)
        x = self.refine_block(x)
        return x

    def forward(self, x):
        x = self.unet(x)
        layer_maps, surface_maps, S = self.branch(x)
        out = self.refinement_module(layer_maps, surface_maps)
        _, refined_surface_maps, refined_S = self.refined_branches(out)

        # S = self.topology_module(S)
        guarantee_S = self.topology_module(refined_S)

        # return S, out, layerProb, mu, x_out, flow
        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'refined_final_surface': refined_S,
                'refined_surface_maps': refined_surface_maps, 'guarantee_surface': guarantee_S}


class FCBRwithMultiscaleRefine(FCBRwithRefine):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        FCBRwithRefine.__init__(
            self, n_layer, input_size, in_channel)
        self.unet = UNet2D_multiscale(n_layer, input_size, in_channel)
        self.branch1 = TwoBranch(n_layer)
        self.branch2 = TwoBranch(n_layer)
        self.branch3 = TwoBranch(n_layer)
        self.branch4 = TwoBranch(n_layer)
        self.multi_branches = [self.branch, self.branch1,
                               self.branch2, self.branch3, self.branch4]

    def forward(self, x):
        layer_maps_list, surface_maps_list, S_list = [], [], []
        x, x_list = self.unet(x)
        for i in range(5):
            layer_maps, surface_maps, S = self.multi_branches[i](x_list[i])
            layer_maps_list.append(layer_maps)
            surface_maps_list.append(surface_maps)
            S_list.append(S)
        out = self.refinement_module(
            layer_maps_list[-1], surface_maps_list[-1])
        _, refined_surface_maps, refined_S = self.branch(out)

        guarantee_S = self.topology_module(refined_S)

        # return S, out, layerProb, mu, x_out, flow
        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'refined_final_surface': refined_S,
                'refined_surface_maps': refined_surface_maps, 'guarantee_surface': guarantee_S,
                'layer_maps_list': layer_maps_list, 'surface_maps_list': surface_maps_list,
                'S_list': S_list}


class UNet2D_connect(UNet2D):

    def __init__(self, n_layer=9, input_size=(128, 1024)):
        super(UNet2D_connect, self).__init__(n_layer, input_size)

    def refinement_module(self, result):
        x = torch.cat([result['surface_maps'], result['layer_maps']], dim=1)

    def forward(self, x):
        result = super(UNet2D_connect, self).forward(x)

        return result


def convert_2d_to_1d(x_2d, dim=3):
    bs = x_2d.shape[0]
    if dim == 2:
        x_1d = x_2d.permute(0, 2, 1, 3)
    else:
        x_1d = x_2d.permute(0, 3, 1, 2)
    x_1d = torch.reshape(
        x_1d, (bs*x_1d.shape[1], x_1d.shape[2], x_1d.shape[3]))

    return x_1d


def convert_1d_to_2d(x_1d, bs, dim=3):
    x_1d = torch.reshape(
        x_1d, (bs, x_1d.shape[0]//bs, x_1d.shape[1], x_1d.shape[2]))
    if dim == 2:
        x_2d = x_1d.permute(0, 2, 1, 3)
    else:
        x_2d = x_1d.permute(0, 2, 3, 1)

    return x_2d


def conv3x1(in_channels, out_channels):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )


def deconv1d(in_channels, out_channels):
    return nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    )


def deconv2d(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    )


class Deconv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = deconv1d(in_ch, out_ch)

    def forward(self, x):
        bs = x.shape[0]
        x = convert_2d_to_1d(x)
        x = self.deconv(x)
        x = convert_1d_to_2d(x, bs)
        return x


class Upsample1D(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.upsample = Bilinear1D(h)

    def forward(self, x):
        bs = x.shape[0]
        x = convert_2d_to_1d(x)
        x = self.upsample(x)
        x = convert_1d_to_2d(x, bs)
        return x


class Upsample2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = Bilinear2D()

    def forward(self, x):
        x = self.upsample(x)
        return x


class ResBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        padding=1,
        activation='PReLU',
        is_1d=True,
    ):
        super().__init__()
        self.is_1d = is_1d
        self.conv1 = conv3x1(in_channels, channels)
        self.conv2 = conv3x1(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        if activation == 'PReLU':
            self.relu1 = nn.PReLU()
            self.relu2 = nn.PReLU()
        else:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
        self.conv3 = conv3x1(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.shortcut = nn.Sequential()

    # def convert_2d_to_1d(self, x_2d, dim=3):
    #     self.bs = x_2d.shape[0]
    #     if dim == 2:
    #         x_1d = x_2d.permute(0, 2, 1, 3)
    #     else:
    #         x_1d = x_2d.permute(0, 3, 1, 2)
    #     x_1d = torch.reshape(
    #         x_1d, (self.bs*x_1d.shape[1], x_1d.shape[2], x_1d.shape[3]))

    #     return x_1d

    # def convert_1d_to_2d(self, x_1d, dim=3):
    #     x_1d = torch.reshape(
    #         x_1d, (self.bs, x_1d.shape[0]//self.bs, x_1d.shape[1], x_1d.shape[2]))
    #     if dim == 2:
    #         x_2d = x_1d.permute(0, 2, 1, 3)
    #     else:
    #         x_2d = x_1d.permute(0, 2, 3, 1)

    #     return x_2d

    def forward(self, x):
        if not self.is_1d:
            bs = x.shape[0]
            x = convert_2d_to_1d(x)
        x = self.conv1(x)
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        if not self.is_1d:
            out = convert_1d_to_2d(out, bs)

        return out


class Bilinear1D(nn.Module):

    def __init__(self, height):
        super().__init__()
        self.upsample = nn.Upsample(
            size=(height, 1), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = x.unsqueeze(3)
        x = self.upsample(x)
        x = x.squeeze(3)

        return x


class Bilinear2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)

        return x


class Maxpooling1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpooling = nn.MaxPool1d(2)

    def forward(self, x):
        bs = x.shape[0]
        x = convert_2d_to_1d(x)
        x = self.maxpooling(x)
        x = convert_1d_to_2d(x, bs)

        return x


class UNet1D(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024)):
        super().__init__()
        # print('UNet1D_init')
        self.n_layer = n_layer

        self.res1_1d = ResBlock1D(1, 64)
        self.res2_1d = ResBlock1D(64, 64)
        self.res3_1d = ResBlock1D(64, 64)
        self.res4_1d = ResBlock1D(64, 64)
        self.res5_1d = ResBlock1D(64, 64)
        self.res6_1d = ResBlock1D(64, 64)

        self.down_res1_1d = ResBlock1D(128, 64)
        self.down_res2_1d = ResBlock1D(128, 64)
        self.down_res3_1d = ResBlock1D(128, 64)
        self.maxpool_1d = nn.MaxPool1d(2)
        self.upsample1_1d = deconv1d(64, 64)
        self.upsample2_1d = deconv1d(64, 64)
        self.upsample3_1d = deconv1d(64, 64)
        self.upsample4_1d = deconv1d(64, 64)
        self.res7_1d = ResBlock1D(128, 64)
        self.res8_1d = ResBlock1D(64, 64)
        self.conv1x1_1d = nn.Conv1d(
            64, 64, kernel_size=1, stride=1, padding=0, bias=False)

    def encoder_1d(self, x):
        x = self.res1_1d(x)
        skip1 = self.res2_1d(x)
        x = self.maxpool_1d(skip1)

        skip2 = self.res3_1d(x)
        x = self.maxpool_1d(skip2)

        skip3 = self.res4_1d(x)
        x = self.maxpool_1d(skip3)

        skip4 = self.res5_1d(x)
        x = self.maxpool_1d(skip4)

        x = self.res6_1d(x)

        return x, skip1, skip2, skip3, skip4

    def decoder_1d(self, x, skip1, skip2, skip3, skip4):
        x = self.upsample1_1d(x)

        x = torch.cat([x, skip4], dim=1)
        x = self.down_res1_1d(x)
        x = self.upsample2_1d(x)

        x = torch.cat([x, skip3], dim=1)
        x = self.down_res2_1d(x)
        x = self.upsample3_1d(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.down_res3_1d(x)
        x = self.upsample4_1d(x)

        x = torch.cat([x, skip1], dim=1)
        x = self.res7_1d(x)
        x = self.res8_1d(x)

        x = self.conv1x1_1d(x)

        return x

    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder_1d(x)
        x = self.decoder_1d(x, skip1, skip2, skip3, skip4)

        return x


class UNet1D_bilinear(UNet1D):

    def __init__(self, n_layer=9, input_size=(128, 1024)):
        super().__init__()
        # print('UNet1D_init')
        self.n_layer = n_layer
        self.upsample1_1d = Bilinear1D(16)
        self.upsample2_1d = Bilinear1D(32)
        self.upsample3_1d = Bilinear1D(64)
        self.upsample4_1d = Bilinear1D(128)

    def decoder_1d(self, x, skip1, skip2, skip3, skip4):
        x = self.upsample1_1d(x)

        x = torch.cat([x, skip4], dim=1)
        x = self.down_res1_1d(x)
        x = self.upsample2_1d(x)

        x = torch.cat([x, skip3], dim=1)
        x = self.down_res2_1d(x)
        x = self.upsample3_1d(x)

        x = torch.cat([x, skip2], dim=1)
        x = self.down_res3_1d(x)
        x = self.upsample4_1d(x)

        x = torch.cat([x, skip1], dim=1)
        x = self.res7_1d(x)
        x = self.res8_1d(x)

        x = self.conv1x1_1d(x)

        return x

    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder_1d(x)
        x = self.decoder_1d(x, skip1, skip2, skip3, skip4)

        return x


class UNet_round(nn.Module):

    def __init__(self, n_layer=9, input_size=(128, 1024)):
        super(UNet_round, self).__init__()
        self.n_layer = n_layer
        self.input_size = input_size

        self.unet1d = UNet1D(n_layer, input_size)

        self.unet2d = [UNet2D(n_layer, input_size) for i in range(n_layer)]

    def forward(self, x):
        xr = x.clone().detach()
        bs = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = torch.reshape(x, (bs*x.shape[1], x.shape[2], x.shape[3]))
        out = self.unet1d(x)
        out = torch.reshape(
            out, (bs, out.shape[0]//bs, out.shape[1], out.shape[2]))
        out = out.permute(0, 2, 3, 1)
        bds = column_wise_softmax(out)
        bds = computeMuVariance(bds)
        bds = torch.round(bds)
        bds = bds.to(torch.int32)
        x = torch.zeros_like(xr)
        for i in range(bs):
            for j in range(self.input_size[1]):
                x[i, 0, :, j] = torch.roll(
                    xr[i, 0, :, j], bds[i, 0, j].item(), dims=0)
        return out


class UNet_1D2D(UNet1D, UNet2D, FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        # UNet1D.__init__(self, n_layer, input_size)
        # UNet2D.__init__(self, n_layer, input_size)
        super().__init__(n_layer, input_size)
        self.bs = 0
        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, channels=(128, 64))

    def convert_2d_to_1d(self, x_2d, dim=3):
        self.bs = x_2d.shape[0]
        if dim == 2:
            x_1d = x_2d.permute(0, 2, 1, 3)
        else:
            x_1d = x_2d.permute(0, 3, 1, 2)
        x_1d = torch.reshape(
            x_1d, (self.bs*x_1d.shape[1], x_1d.shape[2], x_1d.shape[3]))

        return x_1d

    def convert_1d_to_2d(self, x_1d, dim=3):
        x_1d = torch.reshape(
            x_1d, (self.bs, x_1d.shape[0]//self.bs, x_1d.shape[1], x_1d.shape[2]))
        if dim == 2:
            x_2d = x_1d.permute(0, 2, 1, 3)
        else:
            x_2d = x_1d.permute(0, 2, 3, 1)

        return x_2d

    def forward(self, x):
        x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d = self.encoder(x)
        x_1d = self.convert_2d_to_1d(x)
        x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d = self.encoder_1d(x_1d)
        # x_1d = self.convert_1d_to_2d(x_1d)
        # x = torch.cat([x_1d, x_2d], dim=1)
        # x_1d = self.convert_2d_to_1d(x_1d)
        # x_1d = self.res_1d(x_1d)
        x_1d = self.decoder_1d(x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d)
        x_1d = self.convert_1d_to_2d(x_1d)

        # x_2d = self.res_2d(x_2d)
        x_2d = self.decoder(x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d)
        # x_1d = self.convert_1d_to_2d(x_1d)
        x = torch.cat([x_1d, x_2d], dim=1)
        layer_maps, surface_maps, S = self.branch(x)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}


class UNet_1D2D_deconv(UNet1D, UNet2D_deconv, FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        # UNet1D.__init__(self, n_layer, input_size)
        # UNet2D.__init__(self, n_layer, input_size)
        super().__init__(n_layer, input_size)
        self.bs = 0
        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, channels=(128, 64))

    def convert_2d_to_1d(self, x_2d):
        self.bs = x_2d.shape[0]
        x_1d = x_2d.permute(0, 3, 1, 2)
        x_1d = torch.reshape(
            x_1d, (self.bs*x_1d.shape[1], x_1d.shape[2], x_1d.shape[3]))

        return x_1d

    def convert_1d_to_2d(self, x_1d):
        x_1d = torch.reshape(
            x_1d, (self.bs, x_1d.shape[0]//self.bs, x_1d.shape[1], x_1d.shape[2]))
        x_2d = x_1d.permute(0, 2, 3, 1)

        return x_2d

    def forward(self, x):
        x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d = self.encoder(x)
        x_1d = self.convert_2d_to_1d(x)
        x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d = self.encoder_1d(x_1d)
        # x_1d = self.convert_1d_to_2d(x_1d)
        # x = torch.cat([x_1d, x_2d], dim=1)
        # x_1d = self.convert_2d_to_1d(x_1d)
        # x_1d = self.res_1d(x_1d)
        x_1d = self.decoder_1d(x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d)
        x_1d = self.convert_1d_to_2d(x_1d)

        # x_2d = self.res_2d(x_2d)
        x_2d = self.decoder(x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d)
        # x_1d = self.convert_1d_to_2d(x_1d)
        x = torch.cat([x_1d, x_2d], dim=1)
        layer_maps, surface_maps, S = self.branch(x)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}


class UNet_1D2D_bilinear(UNet1D_bilinear, UNet2D, FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1):
        # UNet1D.__init__(self, n_layer, input_size)
        # UNet2D.__init__(self, n_layer, input_size)
        super().__init__(n_layer, input_size)
        self.bs = 0
        self.branch = TwoBranch(n_layer, input_size,
                                use_softargmax=False, channels=(128, 64))

    def convert_2d_to_1d(self, x_2d):
        self.bs = x_2d.shape[0]
        x_1d = x_2d.permute(0, 3, 1, 2)
        x_1d = torch.reshape(
            x_1d, (self.bs*x_1d.shape[1], x_1d.shape[2], x_1d.shape[3]))

        return x_1d

    def convert_1d_to_2d(self, x_1d):
        x_1d = torch.reshape(
            x_1d, (self.bs, x_1d.shape[0]//self.bs, x_1d.shape[1], x_1d.shape[2]))
        x_2d = x_1d.permute(0, 2, 3, 1)

        return x_2d

    def forward(self, x):
        x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d = self.encoder(x)
        x_1d = self.convert_2d_to_1d(x)
        x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d = self.encoder_1d(x_1d)
        # x_1d = self.convert_1d_to_2d(x_1d)
        # x = torch.cat([x_1d, x_2d], dim=1)
        # x_1d = self.convert_2d_to_1d(x_1d)
        # x_1d = self.res_1d(x_1d)
        x_1d = self.decoder_1d(x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d)
        x_1d = self.convert_1d_to_2d(x_1d)

        # x_2d = self.res_2d(x_2d)
        x_2d = self.decoder(x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d)
        # x_1d = self.convert_1d_to_2d(x_1d)
        x = torch.cat([x_1d, x_2d], dim=1)
        layer_maps, surface_maps, S = self.branch(x)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}


class UNet_1D2D_X(UNet_1D2D):
    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        super().__init__(n_layer, input_size)
        self.res_1d = ResBlock1D(128, 64)
        self.res_2d = ResBlock2D(128, 64)
        self.upsample_resize1d = Bilinear1D(height=input_size[1]//16)
        # self.upsample_resize1d = Bilinear1D(height=64)
        self.upsample_resize2d = Bilinear1D(height=input_size[1])
        # self.upsample_resize2d = Bilinear1D(height=1024)
        self.branch = TwoBranch(n_layer, input_size, use_softargmax=False, channels=(
            128, 64), has_edema=has_edema)
        # self.branch_x = TwoBranch(n_layer, input_size,
        #                           use_softargmax=False, channels=(128, 64), has_edema=has_edema)

    def forward(self, x):
        x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d = self.encoder(x)
        x_1d = self.convert_2d_to_1d(x)
        x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d = self.encoder_1d(x_1d)
        x_1d = self.convert_1d_to_2d(x_1d)

        resized_x_1d = self.convert_2d_to_1d(x_1d, dim=2)
        resized_x_1d = self.upsample_resize1d(resized_x_1d)
        resized_x_1d = self.convert_1d_to_2d(resized_x_1d, dim=2)

        resized_x_2d = self.convert_2d_to_1d(x_2d, dim=2)
        resized_x_2d = self.upsample_resize2d(resized_x_2d)
        resized_x_2d = self.convert_1d_to_2d(resized_x_2d, dim=2)

        x_1d_1024 = torch.cat([x_1d, resized_x_2d], dim=1)
        x_2d_64 = torch.cat([resized_x_1d, x_2d], dim=1)
        x_1d = self.convert_2d_to_1d(x_1d_1024)
        x_1d = self.res_1d(x_1d)
        x_1d = self.decoder_1d(x_1d, skip1_1d, skip2_1d, skip3_1d, skip4_1d)
        x_1d = self.convert_1d_to_2d(x_1d)

        x_2d = self.res_2d(x_2d_64)
        x_2d = self.decoder(x_2d, skip1_2d, skip2_2d, skip3_2d, skip4_2d)
        x = torch.cat([x_1d, x_2d], dim=1)
        layer_maps, surface_maps, S = self.branch(x)
        # layer_maps, surface_maps, S = self.branch_x(x)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}


class FCBR1D(FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        FCBR.__init__(self, n_layer, input_size, in_channel, has_edema)
        self.unet1d = UNet1D(n_layer, input_size)
        self.branch_new = TwoBranch(
            n_layer, input_size, use_softargmax=False, channels=(64, 32), has_edema=has_edema)

    def forward(self, x):
        xr = x.clone().detach()
        bs = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = torch.reshape(x, (bs*x.shape[1], x.shape[2], x.shape[3]))
        out = self.unet1d(x)
        out = torch.reshape(
            out, (bs, out.shape[0]//bs, out.shape[1], out.shape[2]))
        out = out.permute(0, 2, 3, 1)
        # layer_maps, surface_maps, S = self.branch(out)
        layer_maps, surface_maps, S = self.branch_new(out)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'guarantee_surface': guarantee_S, 'layer_maps': layer_maps}


class FCBR1DDME(FCBR):

    def __init__(self, n_layer=9, input_size=(128, 1024), in_channel=1, has_edema=False):
        FCBR.__init__(self, n_layer, input_size, in_channel)
        self.unet1d = UNet1D(n_layer, input_size)
        # self.branch_new = TwoBranch(
        #     n_layer, input_size, use_softargmax=False, channels=(64, 32), has_edema=has_edema)

    def forward(self, x):
        xr = x.clone().detach()
        bs = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = torch.reshape(x, (bs*x.shape[1], x.shape[2], x.shape[3]))
        out = self.unet1d(x)
        out = torch.reshape(
            out, (bs, out.shape[0]//bs, out.shape[1], out.shape[2]))
        out = out.permute(0, 2, 3, 1)
        layer_maps, surface_maps, S = self.branch(out)
        # layer_maps, surface_maps, S = self.branch_new(out)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'guarantee_surface': guarantee_S, 'layer_maps': layer_maps}


class Encoder_Block_1D2D_No_Concat(nn.Module):
    def __init__(self, ch=64, in_ty='double'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'first':
            in_ch = 1
        elif in_ty == 'single':
            in_ch = ch
        else:
            in_ch = ch * 2
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch, ch, is_1d=False)
        self.res_1d_3 = ResBlock1D(ch, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch, ch)
        self.res_2d_3 = ResBlock2D(ch, ch)

        self.maxpool_1d = Maxpooling1D()
        self.maxpool_2d = nn.MaxPool2d(2)

    def forward(self, x_1d, x_2d):
        if self.in_ty == 'double':
            x = torch.concat([x_1d, x_2d], dim=1)
        else:
            x = x_1d
        x_1d = self.res_1d_1(x)
        x_1d = self.res_1d_2(x_1d)
        x_1d = self.res_1d_3(x_1d)
        out_1d = self.maxpool_1d(x_1d)

        x_2d = self.res_2d_1(x)
        x_2d = self.res_2d_2(x_2d)
        x_2d = self.res_2d_3(x_2d)
        out_2d = self.maxpool_2d(x_2d)

        return out_1d, out_2d, x_1d, x_2d


class Encoder_Block_1D2D_One_Concat(nn.Module):
    def __init__(self, ch=64, in_ty='double'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'first':
            in_ch = 1
        elif in_ty == 'single':
            in_ch = ch
        else:
            in_ch = ch * 2
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch, ch, is_1d=False)
        self.res_1d_3 = ResBlock1D(ch*2, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch, ch)
        self.res_2d_3 = ResBlock2D(ch*2, ch)

        self.maxpool_1d = Maxpooling1D()
        self.maxpool_2d = nn.MaxPool2d(2)

    def forward(self, x_1d, x_2d):
        if self.in_ty == 'double':
            x = torch.concat([x_1d, x_2d], dim=1)
        else:
            x = x_1d
        x_1d = self.res_1d_1(x)
        x_1d = self.res_1d_2(x_1d)
        x_2d = self.res_2d_1(x)
        x_2d = self.res_2d_2(x_2d)

        x = torch.concat([x_1d, x_2d], dim=1)
        x_1d = self.res_1d_3(x)
        out_1d = self.maxpool_1d(x_1d)
        x_2d = self.res_2d_3(x)
        out_2d = self.maxpool_2d(x_2d)

        return out_1d, out_2d, x_1d, x_2d


class Encoder_Block_1D2D_Two_Concat(nn.Module):
    def __init__(self, ch=64, in_ty='double'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'first':
            in_ch = 1
        elif in_ty == 'single':
            in_ch = ch
        else:
            in_ch = ch * 2
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch*2, ch, is_1d=False)
        self.res_1d_3 = ResBlock1D(ch*2, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch*2, ch)
        self.res_2d_3 = ResBlock2D(ch*2, ch)

        self.maxpool_1d = Maxpooling1D()
        self.maxpool_2d = nn.MaxPool2d(2)

    def forward(self, x_1d, x_2d):
        if self.in_ty == 'double':
            x = torch.concat([x_1d, x_2d], dim=1)
        else:
            x = x_1d
        x_1d = self.res_1d_1(x)
        x_2d = self.res_2d_1(x)

        x = torch.concat([x_1d, x_2d], dim=1)
        x_1d = self.res_1d_2(x)
        x_2d = self.res_2d_2(x)

        x = torch.concat([x_1d, x_2d], dim=1)
        x_1d = self.res_1d_3(x)
        x_2d = self.res_2d_3(x)

        out_1d = self.maxpool_1d(x_1d)
        out_2d = self.maxpool_2d(x_2d)

        return out_1d, out_2d, x_1d, x_2d


class Decoder_Block_1D2D_Simple(nn.Module):
    def __init__(self, ch=64, in_ty='skip'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'single':
            in_ch = ch
        elif in_ty == 'no_skip':
            in_ch = ch*2
        else:
            in_ch = ch*4
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        # self.res_1d_2 = ResBlock1D(ch, ch, is_1d=False)
        # self.res_1d_3 = ResBlock1D(ch, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        # self.res_2d_2 = ResBlock2D(ch, ch)
        # self.res_2d_3 = ResBlock2D(ch, ch)

        self.upsampling_1d = Deconv1D(ch, ch)
        self.upsampling_2d = deconv2d(ch, ch)

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        if self.in_ty == 'single':
            x = out_1d
        elif self.in_ty == 'no_skip':
            x = torch.concat([out_1d, out_2d], dim=1)
        else:
            x = torch.concat([out_1d, out_2d, skip_1d, skip_2d], dim=1)
        x_1d = self.res_1d_1(x)
        # x_1d = self.res_1d_2(x_1d)
        # x_1d = self.res_1d_3(x_1d)

        x_2d = self.res_2d_1(x)
        # x_2d = self.res_2d_2(x_2d)
        # x_2d = self.res_2d_3(x_2d)

        if self.in_ty == 'last':
            out = torch.concat([x_1d, x_2d], dim=1)
            return out, None
        else:
            out_1d = self.upsampling_1d(x_1d)
            out_2d = self.upsampling_2d(x_2d)
            return out_1d, out_2d


class Decoder_Block_1D2D_Bilinear(nn.Module):
    def __init__(self, ch=64, h=128, in_ty='skip'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'single':
            in_ch = ch
        elif in_ty == 'no_skip':
            in_ch = ch*2
        else:
            in_ch = ch*4
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        # self.res_1d_2 = ResBlock1D(ch, ch, is_1d=False)
        # self.res_1d_3 = ResBlock1D(ch, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        # self.res_2d_2 = ResBlock2D(ch, ch)
        # self.res_2d_3 = ResBlock2D(ch, ch)

        self.upsampling_1d = Upsample1D(h)
        self.upsampling_2d = Upsample2D()

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        if self.in_ty == 'single':
            x = out_1d
        elif self.in_ty == 'no_skip':
            x = torch.concat([out_1d, out_2d], dim=1)
        else:
            x = torch.concat([out_1d, out_2d, skip_1d, skip_2d], dim=1)
        x_1d = self.res_1d_1(x)
        # x_1d = self.res_1d_2(x_1d)
        # x_1d = self.res_1d_3(x_1d)

        x_2d = self.res_2d_1(x)
        # x_2d = self.res_2d_2(x_2d)
        # x_2d = self.res_2d_3(x_2d)

        if self.in_ty == 'last':
            out = torch.concat([x_1d, x_2d], dim=1)
            return out, None
        else:
            out_1d = self.upsampling_1d(x_1d)
            out_2d = self.upsampling_2d(x_2d)
            return out_1d, out_2d


class Decoder_Block_1D2D_Bilinear_Two(nn.Module):
    def __init__(self, ch=64, h=128, in_ty='skip'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'single':
            in_ch = ch
        elif in_ty == 'no_skip':
            in_ch = ch*2
        else:
            in_ch = ch*4
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch, ch, is_1d=False)
        # self.res_1d_3 = ResBlock1D(ch, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch, ch)
        # self.res_2d_3 = ResBlock2D(ch, ch)

        self.upsampling_1d = Upsample1D(h)
        self.upsampling_2d = Upsample2D()

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        if self.in_ty == 'single':
            x = out_1d
        elif self.in_ty == 'no_skip':
            x = torch.concat([out_1d, out_2d], dim=1)
        else:
            x = torch.concat([out_1d, out_2d, skip_1d, skip_2d], dim=1)
        x_1d = self.res_1d_1(x)
        x_1d = self.res_1d_2(x_1d)
        # x_1d = self.res_1d_3(x_1d)

        x_2d = self.res_2d_1(x)
        x_2d = self.res_2d_2(x_2d)
        # x_2d = self.res_2d_3(x_2d)

        if self.in_ty == 'last':
            out = torch.concat([x_1d, x_2d], dim=1)
            return out, None
        else:
            out_1d = self.upsampling_1d(x_1d)
            out_2d = self.upsampling_2d(x_2d)
            return out_1d, out_2d


class Decoder_Block_1D2D_No_Concat(nn.Module):
    def __init__(self, ch=64, in_ty='skip'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'single':
            in_ch = ch
        elif in_ty == 'no_skip':
            in_ch = ch*2
        else:
            in_ch = ch*4
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch, ch, is_1d=False)
        # self.res_1d_3 = ResBlock1D(ch, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch, ch)
        # self.res_2d_3 = ResBlock2D(ch, ch)

        self.upsampling_1d = Deconv1D(ch, ch)
        self.upsampling_2d = deconv2d(ch, ch)

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        if self.in_ty == 'single':
            x = out_1d
        elif self.in_ty == 'no_skip':
            x = torch.concat([out_1d, out_2d], dim=1)
        else:
            x = torch.concat([out_1d, out_2d, skip_1d, skip_2d], dim=1)
        x_1d = self.res_1d_1(x)
        x_1d = self.res_1d_2(x_1d)
        # x_1d = self.res_1d_3(x_1d)

        x_2d = self.res_2d_1(x)
        x_2d = self.res_2d_2(x_2d)
        # x_2d = self.res_2d_3(x_2d)

        if self.in_ty == 'last':
            out = torch.concat([x_1d, x_2d], dim=1)
            return out, None
        else:
            out_1d = self.upsampling_1d(x_1d)
            out_2d = self.upsampling_2d(x_2d)
            return out_1d, out_2d


class Decoder_Block_1D2D_One_Concat(nn.Module):
    def __init__(self, ch=64, in_ty='skip'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'single':
            in_ch = ch
        elif in_ty == 'no_skip':
            in_ch = ch*2
        else:
            in_ch = ch*4
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch*2, ch, is_1d=False)
        # self.res_1d_3 = ResBlock1D(ch, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch*2, ch)
        # self.res_2d_3 = ResBlock2D(ch, ch)

        self.upsampling_1d = Deconv1D(ch, ch)
        self.upsampling_2d = deconv2d(ch, ch)

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        if self.in_ty == 'single':
            x = out_1d
        elif self.in_ty == 'no_skip':
            x = torch.concat([out_1d, out_2d], dim=1)
        else:
            x = torch.concat([out_1d, out_2d, skip_1d, skip_2d], dim=1)
        x_1d = self.res_1d_1(x)
        x_2d = self.res_2d_1(x)

        x = torch.concat([x_1d, x_2d], dim=1)
        x_1d = self.res_1d_2(x)
        x_2d = self.res_2d_2(x)
        # x_1d = self.res_1d_3(x_1d)
        # x_2d = self.res_2d_3(x_2d)

        if self.in_ty == 'last':
            out = torch.concat([x_1d, x_2d], dim=1)
            return out, None
        else:
            out_1d = self.upsampling_1d(x_1d)
            out_2d = self.upsampling_2d(x_2d)
            return out_1d, out_2d


class Decoder_Block_1D2D_Two_Concat(nn.Module):
    def __init__(self, ch=64, in_ty='skip'):
        super().__init__()
        self.in_ty = in_ty
        if in_ty == 'single':
            in_ch = ch
        elif in_ty == 'no_skip':
            in_ch = ch*2
        else:
            in_ch = ch*4
        self.res_1d_1 = ResBlock1D(in_ch, ch, is_1d=False)
        self.res_1d_2 = ResBlock1D(ch*2, ch, is_1d=False)
        self.res_1d_3 = ResBlock1D(ch*2, ch, is_1d=False)

        self.res_2d_1 = ResBlock2D(in_ch, ch)
        self.res_2d_2 = ResBlock2D(ch*2, ch)
        self.res_2d_3 = ResBlock2D(ch*2, ch)

        self.upsampling_1d = Deconv1D(ch, ch)
        self.upsampling_2d = deconv2d(ch, ch)

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        if self.in_ty == 'single':
            x = out_1d
        elif self.in_ty == 'no_skip':
            x = torch.concat([out_1d, out_2d], dim=1)
        else:
            x = torch.concat([out_1d, out_2d, skip_1d, skip_2d], dim=1)
        x_1d = self.res_1d_1(x)
        x_2d = self.res_2d_1(x)

        x = torch.concat([x_1d, x_2d], dim=1)
        x_1d = self.res_1d_2(x)
        x_2d = self.res_2d_2(x)
        x = torch.concat([x_1d, x_2d], dim=1)
        x_1d = self.res_1d_3(x)
        x_2d = self.res_2d_3(x)

        if self.in_ty == 'last':
            out = torch.concat([x_1d, x_2d], dim=1)
            return out, None
        else:
            out_1d = self.upsampling_1d(x_1d)
            out_2d = self.upsampling_2d(x_2d)
            return out_1d, out_2d


class Encoder_Block(nn.Module):
    def __init__(self, ch=64, cl_ty='no_connect', in_ty='double'):
        super().__init__()
        if cl_ty == 'no_connect':
            self.block = Encoder_Block_1D2D_No_Concat(ch, in_ty)
        elif cl_ty == 'bilinear_no_connect':
            self.block = Encoder_Block_1D2D_No_Concat(ch, in_ty)
        elif cl_ty == 'bilinear_two_no_connect':
            self.block = Encoder_Block_1D2D_No_Concat(ch, in_ty)
        elif cl_ty == 'simple_no_connect':
            self.block = Encoder_Block_1D2D_No_Concat(ch, in_ty)
        elif cl_ty == 'one_connect':
            self.block = Encoder_Block_1D2D_One_Concat(ch, in_ty)
        elif cl_ty == 'two_connect':
            self.block = Encoder_Block_1D2D_Two_Concat(ch, in_ty)
        elif cl_ty == 'simple_two_connect':
            self.block = Encoder_Block_1D2D_Two_Concat(ch, in_ty)
        elif cl_ty == 'bilinear_two_connect':
            self.block = Encoder_Block_1D2D_Two_Concat(ch, in_ty)
        elif cl_ty == 'all_connect':
            self.block = Encoder_Block_1D2D_Two_Concat(ch, in_ty)
        elif cl_ty == 'two_two_connect':
            self.block = Encoder_Block_1D2D_Two_Concat(ch, in_ty)

    def forward(self, x_1d, x_2d):
        out_1d, out_2d, skip_1d, skip_2d = self.block(x_1d, x_2d)

        return out_1d, out_2d, skip_1d, skip_2d


class Decoder_Block(nn.Module):
    def __init__(self, ch=64, cl_ty='no_connect', h=128, in_ty='skip'):
        super().__init__()
        if cl_ty == 'no_connect':
            self.block = Decoder_Block_1D2D_No_Concat(ch, in_ty)
        elif cl_ty == 'simple_no_connect':
            self.block = Decoder_Block_1D2D_Simple(ch, in_ty)
        elif cl_ty == 'simple_two_connect':
            self.block = Decoder_Block_1D2D_Simple(ch, in_ty)
        elif cl_ty == 'all_connect':
            self.block = Decoder_Block_1D2D_One_Concat(ch, in_ty)
        elif cl_ty == 'bilinear_no_connect':
            self.block = Decoder_Block_1D2D_Bilinear(ch, h, in_ty)
        elif cl_ty == 'bilinear_two_connect':
            self.block = Decoder_Block_1D2D_Bilinear(ch, h, in_ty)
        elif cl_ty == 'bilinear_two_no_connect':
            self.block = Decoder_Block_1D2D_Bilinear_Two(ch, h, in_ty)
        elif cl_ty == 'two_two_connect':
            self.block = Decoder_Block_1D2D_Two_Concat(ch, in_ty)

    def forward(self, out_1d, out_2d, skip_1d, skip_2d):
        out_1d, out_2d = self.block(out_1d, out_2d, skip_1d, skip_2d)

        return out_1d, out_2d


class UNet_1D2D_Plus(UNet_1D2D):
    def __init__(self, n_layer=9, input_size=(128, 1024), cl_ty='no_connect'):
        super().__init__(n_layer, input_size)
        ch = 64

        self.en_1_1 = Encoder_Block(ch, cl_ty, 'first')
        self.en_2_1 = Encoder_Block(ch, cl_ty, 'single')
        self.en_2_2 = Encoder_Block(ch, cl_ty, 'single')
        self.en_3_1 = Encoder_Block(ch, cl_ty, 'single')
        self.en_3_2 = Encoder_Block(ch, cl_ty)
        self.en_3_3 = Encoder_Block(ch, cl_ty, 'single')
        self.en_4_1 = Encoder_Block(ch, cl_ty, 'single')
        self.en_4_2 = Encoder_Block(ch, cl_ty)
        self.en_4_3 = Encoder_Block(ch, cl_ty)
        self.en_4_4 = Encoder_Block(ch, cl_ty, 'single')

        self.de_1_1 = Decoder_Block(ch, cl_ty, 0, 'last')
        self.de_2_1 = Decoder_Block(ch, cl_ty, 128)
        self.de_2_2 = Decoder_Block(ch, cl_ty, 128)
        self.de_3_1 = Decoder_Block(ch, cl_ty, 64)
        self.de_3_2 = Decoder_Block(ch, cl_ty, 64)
        self.de_3_3 = Decoder_Block(ch, cl_ty, 64)
        self.de_4_1 = Decoder_Block(ch, cl_ty, 32)
        self.de_4_2 = Decoder_Block(ch, cl_ty, 32)
        self.de_4_3 = Decoder_Block(ch, cl_ty, 32)
        self.de_4_4 = Decoder_Block(ch, cl_ty, 32)
        self.de_5_1 = Decoder_Block(ch, cl_ty, 16, 'single')
        self.de_5_2 = Decoder_Block(ch, cl_ty, 16, 'no_skip')
        self.de_5_3 = Decoder_Block(ch, cl_ty, 16, 'no_skip')
        self.de_5_4 = Decoder_Block(ch, cl_ty, 16, 'no_skip')
        self.de_5_5 = Decoder_Block(ch, cl_ty, 16, 'single')

    def forward(self, x):
        en_64_1024_1d, en_64_512_2d, sk_128_1024_1d, sk_128_1024_2d = self.en_1_1(
            x, None)
        en_32_512_1d, en_32_256_2d, sk_64_512_1d, sk_64_512_2d = self.en_2_1(
            en_64_512_2d, None)
        en_32_1024_1d, en_32_512_2d, sk_64_1024_1d, sk_64_1024_2d = self.en_2_2(
            en_64_1024_1d, None)
        en_16_256_1d, en_16_128_2d, sk_32_256_1d, sk_32_256_2d = self.en_3_1(
            en_32_256_2d, None)
        en_16_512_1d, en_16_256_2d, sk_32_512_1d, sk_32_512_2d = self.en_3_2(
            en_32_512_1d, en_32_512_2d)
        en_16_1024_1d, en_16_512_2d, sk_32_1024_1d, sk_32_1024_2d = self.en_3_3(
            en_32_1024_1d, None)
        en_8_128_1d, en_8_64_2d, sk_16_128_1d, sk_16_128_2d = self.en_4_1(
            en_16_128_2d, None)
        en_8_256_1d, en_8_128_2d, sk_16_256_1d, sk_16_256_2d = self.en_4_2(
            en_16_256_1d, en_16_256_2d)
        en_8_512_1d, en_8_256_2d, sk_16_512_1d, sk_16_512_2d = self.en_4_3(
            en_16_512_1d, en_16_512_2d)
        en_8_1024_1d, en_8_512_2d, sk_16_1024_1d, sk_16_1024_2d = self.en_4_4(
            en_16_1024_1d, None)

        _, de_16_128_2d = self.de_5_1(en_8_64_2d, None, None, None)
        de_16_128_1d, de_16_256_2d = self.de_5_2(
            en_8_128_1d, en_8_128_2d, None, None)
        de_16_256_1d, de_16_512_2d = self.de_5_3(
            en_8_256_1d, en_8_256_2d, None, None)
        de_16_512_1d, de_16_1024_2d = self.de_5_4(
            en_8_512_1d, en_8_512_2d, None, None)
        de_16_1024_1d, _ = self.de_5_5(en_8_1024_1d, None, None, None)
        _, de_32_256_2d = self.de_4_1(
            de_16_128_1d, de_16_128_2d, sk_16_128_1d, sk_16_128_2d)
        de_32_256_1d, de_32_512_2d = self.de_4_2(
            de_16_256_1d, de_16_256_2d, sk_16_256_1d, sk_16_256_2d)
        de_32_512_1d, de_32_1024_2d = self.de_4_3(
            de_16_512_1d, de_16_512_2d, sk_16_512_1d, sk_16_512_2d)
        de_32_1024_1d, _ = self.de_4_4(
            de_16_1024_1d, de_16_1024_2d, sk_16_1024_1d, sk_16_1024_2d)
        _, de_64_512_2d = self.de_3_1(
            de_32_256_1d, de_32_256_2d, sk_32_256_1d, sk_32_256_2d)
        de_64_512_1d, de_64_1024_2d = self.de_3_2(
            de_32_512_1d, de_32_512_2d, sk_32_512_1d, sk_32_512_2d)
        de_64_1024_1d, _ = self.de_3_3(
            de_32_1024_1d, de_32_1024_2d, sk_32_1024_1d, sk_32_1024_2d)
        _, de_128_1024_2d = self.de_2_1(
            de_64_512_1d, de_64_512_2d, sk_64_512_1d, sk_64_512_2d)
        de_128_1024_1d, _ = self.de_2_2(
            de_64_1024_1d, de_64_1024_2d, sk_64_1024_1d, sk_64_1024_2d)
        out, _ = self.de_1_1(de_128_1024_1d, de_128_1024_2d,
                             sk_128_1024_1d, sk_128_1024_2d)

        layer_maps, surface_maps, S = self.branch(out)
        guarantee_S = self.topology_module(S)

        return {'final_surfaces': S, 'surface_maps': surface_maps,
                'layer_maps': layer_maps, 'guarantee_surface': guarantee_S}


if __name__ == '__main__':
    model = UNet2D(n_layer=9, input_size=(128, 1024))
    model.to('cuda:0')
    summary(model=model, input_size=(2, 1, 128, 1024))
    # a = torch.tensor([[1, 2, 1], [4, 3, 5], [6, 1, 2]])
    # y_max, x_max = softargmax_2D(a)
    # print(x_max)
    # print(y_max)

    # beta = 100
    # y_est = np.array([[1.1, 3.0, 1.1, 3.1, 0.8]])
    # a = np.exp(beta*y_est)
    # # b = np.sum(np.exp(beta*y_est))
    # b = np.sum(a)
    # softmax = a/b
    # max = np.sum(softmax*y_est)
    # print(max)
    # pos = range(y_est.size)
    # softargmax = np.sum(softmax*pos)
    # print(softargmax)

    # beta = 100
    # y_est = np.array([[1.1, 3.0, 0.5],
    #                   [1.2, 0.3, 0.2],
    #                   [1.0, 2.5, 1.5],
    #                   [0.5, 0.5, 0.5],
    #                   [0.4, 1.5, 1.7], ])
    # a = np.exp(beta*y_est)
    # # b = np.sum(np.exp(beta*y_est))
    # b = np.sum(a, 0)
    # softmax = a/b
    # max = np.sum(softmax*y_est, 0)
    # print(max)
    # pos = np.zeros_like(y_est)
    # for i in range(pos.shape(0)):
    #     pos[i, :] = i
    # softargmax = np.sum(softmax*pos, 0)
    # print(softargmax)

    # a = torch.rand([2, 2, 4, 3])
    # print(a)
    # a = torch.tensor([[
    #     [[1.1, 3.0, 0.5],
    #      [0.0, 2.0, 0.6],
    #      [1.2, 0.3, 0.2], ],

    #     [[0.5, 0.5, 0.5],
    #      [0.3, 0.5, 0.5],
    #      [0.4, 1.5, 1.7], ], ],

    #     [[[1.1, 1.0, 0.5],
    #       [1.0, 0.9, 0.4],
    #       [1.0, 0.3, 0.2], ],

    #      [[0.5, 0.5, 0.5],
    #       [0.4, 0.5, 0.5],
    #       [0.4, 1.5, 1.5], ], ],
    # ])
    # print(a.shape)
    # b = softargmax(a, beta=40)
    # c = torch.round(b, decimals=2)
    # # b = b.to(torch.int32)
    # print(c)
