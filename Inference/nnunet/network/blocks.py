import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim='3d',
                 grn=False
                 ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='3d', grn=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim,
                         grn=grn)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, norm_type='group', dim='3d', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim,
                         grn=grn)

        self.resample_do_res = do_res

        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class AttBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, patch_size):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(AttBlock, self).__init__()

        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, padding=1)
        self.norm1 = nn.InstanceNorm3d(output_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.lr = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.ratio = 8

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fc1 = nn.Conv3d(output_channels, output_channels // self.ratio, 1, bias=False)
        self.fc2 = nn.Conv3d(output_channels // self.ratio, output_channels, 1, bias=False)

        self.conv_d_3 = nn.Conv3d(output_channels, 1, (3, 1, 1), padding=(1, 0, 0))
        self.conv_d_7 = nn.Conv3d(output_channels, 1, (7, 1, 1), padding=(3, 0, 0))
        self.norm_d = nn.InstanceNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.glu_d = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc1_d = nn.Conv3d(1, 1, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.fc2_d = nn.Conv3d(1, 1, (3, 1, 1), padding=(1, 0, 0), bias=False)

        self.conv_h_3 = nn.Conv3d(output_channels, 1, (1, 3, 1), padding=(0, 1, 0))
        self.conv_h_7 = nn.Conv3d(output_channels, 1, (1, 7, 1), padding=(0, 3, 0))
        self.norm_h = nn.InstanceNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.glu_h = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc1_h = nn.Conv3d(1, 1, (1, 3, 1), padding=(0, 1, 0), bias=False)
        self.fc2_h = nn.Conv3d(1, 1, (1, 3, 1), padding=(0, 1, 0), bias=False)

        self.conv_w_3 = nn.Conv3d(output_channels, 1, (1, 1, 3), padding=(0, 0, 1))
        self.conv_w_7 = nn.Conv3d(output_channels, 1, (1, 1, 7), padding=(0, 0, 3))
        self.norm_w = nn.InstanceNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.glu_w = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc1_w = nn.Conv3d(1, 1, (1, 1, 3), padding=(0, 0, 1), bias=False)
        self.fc2_w = nn.Conv3d(1, 1, (1, 1, 3), padding=(0, 0, 1), bias=False)

        self.h_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.h_max_pool = nn.AdaptiveMaxPool2d(1)
        self.w_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.w_max_pool = nn.AdaptiveMaxPool2d(1)
        self.d_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.d_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # print(x.shape)
        # print(self.patch_size)
        x = self.lr(self.norm1(self.conv1(x)))
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        ca_out = avg_out + max_out
        ca_out = torch.sigmoid(ca_out)

        x = ca_out * x + x  # B C D H W

        # D view
        x_d_3 = self.conv_d_3(x)
        x_d_7 = self.conv_d_7(x)
        x_d = x_d_3 + x_d_7
        x_d = self.glu_d(self.norm_d(x_d))
        sa_out_avg_d = self.d_avg_pool(x_d)  # B 1 D 1 1
        sa_out_avg_d = self.fc2_d(self.activation(self.fc1_d(sa_out_avg_d)))
        x_d = x_d.squeeze(1)
        sa_out_max_d = self.d_max_pool(x_d)
        sa_out_max_d = sa_out_max_d.unsqueeze(1)
        sa_out_max_d = self.fc2_d(self.activation(self.fc1_d(sa_out_max_d)))
        sa_out_d = sa_out_avg_d + sa_out_max_d
        sa_out_d = torch.sigmoid(sa_out_d)
        x1 = sa_out_d * x + x

        # H view
        x_h_3 = self.conv_h_3(x)
        x_h_7 = self.conv_h_7(x)
        x_h = x_h_3 + x_h_7
        x_h = self.glu_h(self.norm_h(x_h))
        sa_out_avg_h = self.d_avg_pool(x_h)  # B 1 D 1 1
        sa_out_avg_h = self.fc2_h(self.activation(self.fc1_h(sa_out_avg_h)))
        x_h = x_h.squeeze(1)
        sa_out_max_h = self.d_max_pool(x_h)
        sa_out_max_h = sa_out_max_h.unsqueeze(1)
        sa_out_max_h = self.fc2_h(self.activation(self.fc1_h(sa_out_max_h)))
        sa_out_h = sa_out_avg_h + sa_out_max_h
        sa_out_h = torch.sigmoid(sa_out_h)
        x2 = sa_out_h * x + x

        # W view
        x_w_3 = self.conv_w_3(x)
        x_w_7 = self.conv_w_7(x)
        x_w = x_w_3 + x_w_7
        x_w = self.glu_d(self.norm_d(x_w))
        sa_out_avg_w = self.d_avg_pool(x_w)  # B 1 D 1 1
        sa_out_avg_w = self.fc2_w(self.activation(self.fc1_w(sa_out_avg_w)))
        x_w = x_w.squeeze(1)
        sa_out_max_w = self.d_max_pool(x_w)
        sa_out_max_w = sa_out_max_w.unsqueeze(1)
        sa_out_max_w = self.fc2_w(self.activation(self.fc1_w(sa_out_max_w)))
        sa_out_w = sa_out_avg_w + sa_out_max_w
        sa_out_w = torch.sigmoid(sa_out_w)
        x3 = sa_out_w * x + x

        return [x, x1, x2, x3]


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, stage=0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.trained_D, self.trained_H, self.trained_W = input_resolution
        self.trained_len = self.trained_D * self.trained_H * self.trained_W
        self.trained_pool_D, self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, \
                                                                        input_resolution[
                                                                            1] // self.sr_ratio, input_resolution[
                                                                            2] // self.sr_ratio
        self.trained_pool_len = self.trained_pool_D * self.trained_pool_H * self.trained_pool_W
        if stage == 3:
            self.local_len = None
        else:
            self.local_len = window_size ** 2
            self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
            self.learnable_tokens_w = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(num_heads, self.head_dim * self.trained_W, self.local_len), mean=0,
                    std=0.02))
            self.learnable_tokens_h = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(num_heads, self.head_dim * self.trained_H, self.local_len), mean=0,
                    std=0.02))
            self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))
        self.num_heads = num_heads

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        #     self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.stage = stage
        self.dwconv1 = nn.Conv3d(dim, dim, kernel_size=3,
                                 stride=1, padding=1, groups=dim, bias=True)
        self.dwconv2 = nn.Conv3d(dim, dim, kernel_size=3,
                                 stride=1, padding=1, groups=dim, bias=True)
        self.dwconv3 = nn.Conv3d(dim, dim, kernel_size=3,
                                 stride=1, padding=1, groups=dim, bias=True)
        self.project_out_q1 = nn.InstanceNorm3d(dim)
        self.project_out_k1 = nn.InstanceNorm3d(dim)
        self.project_out_v = nn.InstanceNorm3d(dim)
        self.project_out_k2 = nn.InstanceNorm3d(dim)
        self.project_out_q2 = nn.InstanceNorm3d(dim)

        self.conv0_1 = nn.Conv3d(dim, dim, (1, 3, 1), padding=(0, 1, 0), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (1, 3, 1), padding=(0, 1, 0), groups=dim)
        self.conv0_3 = nn.Conv3d(dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)
        self.conv0_4 = nn.Conv3d(dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.conv1_3 = nn.Conv3d(dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)
        self.conv1_4 = nn.Conv3d(dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)

        self.conv2_1 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.conv2_2 = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.conv2_3 = nn.Conv3d(dim, dim, (1, 3, 1), padding=(0, 1, 0), groups=dim)
        self.conv2_4 = nn.Conv3d(dim, dim, (1, 3, 1), padding=(0, 1, 0), groups=dim)

        # self.conv1 = nn.Conv3d(dim * 3, dim, (3, 3, 3), padding=(1, 1, 1), groups=dim)
        self.conv1 = nn.Conv3d(dim * 2, dim, (3, 3, 3), padding=(1, 1, 1), groups=dim)

        self.cpb_fc1 = nn.Linear(3, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, 9), mean=0,
                                  std=0.0004))
        self.act = nn.GELU()
        self.sr = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x0 = self.dwconv1(x[0])
        x1 = self.dwconv1(x[1])
        x2 = self.dwconv1(x[2])

        b, c, d, h, w = x0.shape
        attn_01 = self.conv0_1(x0)
        attn_01 = self.conv0_3(attn_01)  # enhance D and HW
        attn_02 = self.conv0_4(x0)
        attn_02 = self.conv0_2(attn_02)  # enhance D and WH

        attn_11 = self.conv1_1(x1)
        attn_11 = self.conv1_3(attn_11)  # enhance H and DW
        attn_12 = self.conv1_4(x1)
        attn_12 = self.conv1_2(attn_12)  # enhance H and WD

        attn_21 = self.conv2_1(x2)
        attn_21 = self.conv2_3(attn_21)  # enhance W and DH
        attn_22 = self.conv2_4(x2)
        attn_22 = self.conv2_2(attn_22)  # enhance W and HD
        #
        out1 = attn_01 + attn_02  # enhance D 横断面
        out2 = attn_11 + attn_12  # enhance H 冠状面
        out3 = attn_21 + attn_22  # enhance W 矢状面
        out_q1 = self.project_out_q1(out3)
        out_k1 = self.project_out_k1(out1)
        out_v = self.project_out_v(out1)
        out_k2 = self.project_out_k2(out1)
        out_q2 = self.project_out_q2(out2)

        q1 = rearrange(out_q1, 'b (head c) d h w -> b head (d h) (w c)', head=self.num_heads)
        k1 = rearrange(out_k1, 'b (head c) d h w -> b head (d h) (w c)', head=self.num_heads)
        v1 = rearrange(out_v, 'b (head c) d h w -> b head (d h) (w c)', head=self.num_heads)
        k2 = rearrange(out_k2, 'b (head c) d h w -> b head (d w) (h c)', head=self.num_heads)
        v2 = rearrange(out_v, 'b (head c) d h w -> b head (d w) (h c)', head=self.num_heads)
        q2 = rearrange(out_q2, 'b (head c) d h w -> b head (d w) (h c)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        o1 = (attn1 @ v1)
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        o2 = (attn2 @ v2)
        o1 = rearrange(o1, 'b head (d h) (w c) -> b (head c) d h w', head=self.num_heads, h=h, w=w, d=d)
        o2 = rearrange(o2, 'b head (d w) (h c) -> b (head c) d h w', head=self.num_heads, h=h, w=w, d=d)
        o1 = o1 + out_v
        o2 = o2 + out_v
        out = torch.cat([o1, o2], dim=1)
        out = self.conv1(out)

        return out


class InterTransBlock(nn.Module):
    def __init__(self, dim, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., num_heads=8,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, stage=0):
        super(InterTransBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_3 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_4 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = Attention(dim, input_resolution, num_heads=num_heads, window_size=window_size,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   sr_ratio=sr_ratio, stage=stage)
        self.FFN = MLP(dim)

    def forward(self, x):
        x_pre = x[0]
        x1 = x[1].permute(0, 2, 3, 4, 1)  # enhance D
        x2 = x[2].permute(0, 2, 3, 4, 1)  # enhance H
        x3 = x[3].permute(0, 2, 3, 4, 1)  # enhance W

        x1 = self.SlayerNorm_1(x1)
        x2 = self.SlayerNorm_2(x2)
        x3 = self.SlayerNorm_3(x3)
        # print(x.shape)
        x1 = x1.permute(0, 4, 1, 2, 3)
        x2 = x2.permute(0, 4, 1, 2, 3)
        x3 = x3.permute(0, 4, 1, 2, 3)
        # print(x.shape)
        x = self.Attention([x1, x2, x3])  # padding 到right_size
        x = x + x_pre

        x = x.permute(0, 2, 3, 4, 1)
        h = x
        x = self.SlayerNorm_4(x)
        # print(x.shape)
        x = self.FFN(x)
        x = h + x
        x = x.permute(0, 4, 1, 2, 3)

        return x


if __name__ == "__main__":
    # network = nnUNeXtBlock(in_channels=12, out_channels=12, do_res=False).cuda()

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 8, 8, 8)).cuda()
    #     print(network(x).shape)

    # network = DownsampleBlock(in_channels=12, out_channels=24, do_res=False)

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 128, 128, 128))
    #     print(network(x).shape)

    network = MedNeXtBlock(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').cuda()
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 12, 64, 64, 64)).cuda()
        print(network(x).shape)
